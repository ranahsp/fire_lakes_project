"""
Rolling-origin (expanding-window) robustness check for the LSTM CHLA
fire-ablation result.

Tests whether the "fire increases CHLA RMSE" direction holds at
different points in time, not just the original 2021-2022 test window.
Three folds, each with its own chronological train/val/test split and
its own normalization statistics computed ONLY from that fold's own
training years (the original script hardcoded 2002-2018 for
normalization regardless of split -- this version fixes that, since
using it unmodified would leak future years into Fold 1/2 normalization).

Usage: python3 lstm_ablation_rolling_origin.py <variable> <horizon> <fold_id>
  fold_id: 1, 2, or 3
    Fold 1: train 2002-2014, val 2015-2016, test 2017-2018
    Fold 2: train 2002-2016, val 2017-2018, test 2019-2020
    Fold 3: train 2002-2018, val 2019-2020, test 2021-2022 (= original result)

All outputs (cache, checkpoints, results) are isolated per-fold, so this
NEVER reads or writes the original lstm_ablation.py's files.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR       = "/u/home/elwasila/firelakes/data/model_ready"

VARIABLE = sys.argv[1] if len(sys.argv) > 1 else 'chla'
TARGET_HORIZON = sys.argv[2] if len(sys.argv) > 2 else '1week'
FOLD_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 1
assert VARIABLE in ['chla', 'tsm']
assert TARGET_HORIZON in ['1week', '1month', '3month']
assert FOLD_ID in [1, 2, 3]

FOLD_DEFINITIONS = {
    1: {'train': (2002, 2014), 'val': (2015, 2016), 'test': (2017, 2018)},
    2: {'train': (2002, 2016), 'val': (2017, 2018), 'test': (2019, 2020)},
    3: {'train': (2002, 2018), 'val': (2019, 2020), 'test': (2021, 2022)},
}
FOLD = FOLD_DEFINITIONS[FOLD_ID]

# Isolated per-fold paths -- never touches the original script's files
RESULTS_DIR    = f"/u/home/elwasila/firelakes/models/results/LSTM/rolling_origin_fold{FOLD_ID}"
CHECKPOINT_DIR = f"/u/home/elwasila/firelakes/models/checkpoints/rolling_origin_fold{FOLD_ID}"
SEQ_CACHE_DIR  = f"/u/home/elwasila/firelakes/data/model_ready/seq_cache/rolling_origin_fold{FOLD_ID}"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SEQ_CACHE_DIR, exist_ok=True)

TARGET_BASE = f'{VARIABLE}_next_{TARGET_HORIZON}'
TARGET_ANOMALY = f'{TARGET_BASE}_anomaly'

LOOKBACK    = 10
BATCH_SIZE  = 256
HIDDEN_SIZE = 96
EMBED_DIM   = 16
DROPOUT     = 0.3
LR          = 0.0001
MAX_EPOCHS  = 50
PATIENCE    = 5
GRAD_CLIP   = 1.0
NEUTRAL_VALUE = 0.5
SEEDS       = [42, 123, 2024]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Rolling-origin Fold {FOLD_ID} | Variable: {VARIABLE} | Horizon: {TARGET_HORIZON}", flush=True)
print(f"  Train: {FOLD['train']} | Val: {FOLD['val']} | Test: {FOLD['test']}", flush=True)
print(f"Using device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

DYNAMIC_COLS_BASE = [
    'chla_anomaly', 'tsm_anomaly', 'lswt_anomaly',
    'temp_2m_mean_c_anomaly', 'temp_2m_max_c_anomaly',
    'precip_mm_anomaly', 'runoff_mm_anomaly',
    'surface_runoff_mm_anomaly', 'wind_speed_ms_anomaly',
    'radiation_wm2_anomaly',
]
FIRE_COLS = [
    'ba_weekly_km2', 'ba_prop_catchment',
    'ba_forest_km2', 'ba_grassland_km2',
    'ba_shrubland_km2', 'ba_cropland_km2',
    'days_since_fire',
    'n_fire_events', 'max_fire_size_km2', 'total_perimeter_km',
]
STATIC_COLS = [
    'depth_avg_m', 'vol_total_km3', 'wshd_area_km2',
    'res_time_days', 'elevation_m', 'lc_tre_usu', 'slp_dg_uav',
]

def safe_minmax(arr):
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (0.0, 1.0, True)
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return (vmin, vmax, True)
    return (vmin, vmax, False)

def build_dataset(version_name, dynamic_cols):
    cache_path = os.path.join(SEQ_CACHE_DIR, f'{version_name}_{VARIABLE}_{TARGET_HORIZON}_L{LOOKBACK}.pkl')
    if os.path.exists(cache_path):
        print(f"Loading cached sequences: {cache_path}", flush=True)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Building sequences for {version_name} (Fold {FOLD_ID})...", flush=True)
    train_df = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_train.parquet'))
    val_df   = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_val.parquet'))
    test_df  = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_test.parquet'))
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(['lake_name', 'cell_id', 'year', 'week'])
    del train_df, val_df, test_df

    with open(os.path.join(DATA_DIR, f'{VARIABLE}_lake_weights_v2.json')) as f:
        lake_weights = json.load(f)

    lake_names = sorted(full_df['lake_name'].unique())
    lake_to_idx = {name: i for i, name in enumerate(lake_names)}

    ALL_NORM_COLS = dynamic_cols + STATIC_COLS + [TARGET_ANOMALY]
    norm_stats = {}
    # FOLD-AWARE: normalization uses ONLY this fold's own training years,
    # not the hardcoded 2002-2018 range from the original script
    train_sy, train_ey = FOLD['train']
    train_mask = (full_df['year'] >= train_sy) & (full_df['year'] <= train_ey)
    for lake in lake_names:
        lt = full_df[train_mask & (full_df['lake_name'] == lake)]
        if len(lt) == 0:
            continue
        stats = {}
        for col in ALL_NORM_COLS:
            vmin, vmax, degenerate = safe_minmax(lt[col].values.astype(float))
            stats[col] = (vmin, vmax, degenerate)
        norm_stats[lake] = stats

    def normalize(value, lake, col):
        if lake not in norm_stats or col not in norm_stats[lake]:
            return value
        vmin, vmax, degenerate = norm_stats[lake][col]
        if degenerate:
            if np.isscalar(value):
                return NEUTRAL_VALUE
            return np.full_like(np.asarray(value, dtype=np.float32), NEUTRAL_VALUE)
        return (value - vmin) / (vmax - vmin)

    # FOLD-AWARE splits, matching FOLD_DEFINITIONS exactly
    splits = {'train': FOLD['train'], 'val': FOLD['val'], 'test': FOLD['test']}
    split_samples = {'train': [], 'val': [], 'test': []}

    for (lake, cell_id), group in full_df.groupby(['lake_name', 'cell_id'], observed=True):
        group = group.sort_values(['year', 'week']).reset_index(drop=True)
        if lake not in norm_stats:
            continue

        feat_matrix = np.zeros((len(group), len(dynamic_cols)), dtype=np.float32)
        for j, col in enumerate(dynamic_cols):
            feat_matrix[:, j] = normalize(group[col].values, lake, col).astype(np.float32)
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if TARGET_ANOMALY not in group.columns:
            continue
        target_norm = normalize(group[TARGET_ANOMALY].values, lake, TARGET_ANOMALY).astype(np.float32)

        static_vals_raw = group[STATIC_COLS].iloc[0].fillna(0)
        static_vals = np.array([
            normalize(static_vals_raw[col], lake, col) for col in STATIC_COLS
        ], dtype=np.float32)
        static_vals = np.nan_to_num(static_vals, nan=0.0, posinf=0.0, neginf=0.0)

        lake_idx = lake_to_idx[lake]
        weight = lake_weights.get(lake, 1.0)
        years = group['year'].values

        for t in range(LOOKBACK, len(group)):
            target_val = group[TARGET_ANOMALY].iloc[t]
            if pd.isna(target_val):
                continue
            row_year = years[t]
            for split_name, (sy, ey) in splits.items():
                if sy <= row_year <= ey:
                    seq = feat_matrix[t-LOOKBACK:t]
                    split_samples[split_name].append((
                        seq, static_vals, lake_idx, target_norm[t], weight
                    ))
                    break

    del full_df
    result = {'splits': split_samples, 'lake_to_idx': lake_to_idx,
              'norm_stats': norm_stats, 'n_lakes': len(lake_names)}
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Cached to: {cache_path}", flush=True)
    return result

class SeqDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        seq, static_vals, lake_idx, target, weight = self.samples[idx]
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(static_vals, dtype=torch.float32),
                torch.tensor(lake_idx, dtype=torch.long),
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(weight, dtype=torch.float32))

class FireLakeLSTM(nn.Module):
    def __init__(self, n_dynamic, n_static, n_lakes, hidden_size=96, embed_dim=16, dropout=0.3):
        super().__init__()
        self.lake_embed = nn.Embedding(n_lakes, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.lstm = nn.LSTM(input_size=n_dynamic, hidden_size=hidden_size,
                             num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + embed_dim + n_static, 64),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, seq, static_vals, lake_idx):
        lstm_out, (h_n, c_n) = self.lstm(seq)
        last_hidden = h_n[-1]
        lake_emb = self.embed_norm(self.lake_embed(lake_idx))
        combined = torch.cat([last_hidden, lake_emb, static_vals], dim=1)
        return self.fc(combined).squeeze(-1)

def weighted_mse(pred, target, weight):
    return ((pred - target)**2 * weight).mean()

def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)

def run_single_seed(version_name, dynamic_cols, data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = SeqDataset(data['splits']['train'])
    val_ds   = SeqDataset(data['splits']['val'])
    test_ds  = SeqDataset(data['splits']['test'])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FireLakeLSTM(len(dynamic_cols), len(STATIC_COLS), data['n_lakes'],
                          HIDDEN_SIZE, EMBED_DIM, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'lstm_{version_name}_{VARIABLE}_{TARGET_HORIZON}_seed{seed}.pt')
    start_epoch, best_val_loss, patience_counter = 0, float('inf'), 0
    best_state = model.state_dict()

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        patience_counter = ckpt['patience_counter']
        best_state = ckpt['best_model_state']
        print(f"  [seed {seed}] Resuming from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        n_train = 0
        for seq, static_vals, lake_idx, target, weight in train_loader:
            seq, static_vals = seq.to(device), static_vals.to(device)
            lake_idx, target, weight = lake_idx.to(device), target.to(device), weight.to(device)
            optimizer.zero_grad()
            pred = model(seq, static_vals, lake_idx)
            loss = weighted_mse(pred, target, weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * len(target)
            n_train += len(target)
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for seq, static_vals, lake_idx, target, weight in val_loader:
                seq, static_vals = seq.to(device), static_vals.to(device)
                lake_idx, target, weight = lake_idx.to(device), target.to(device), weight.to(device)
                pred = model(seq, static_vals, lake_idx)
                loss = weighted_mse(pred, target, weight)
                val_loss += loss.item() * len(target)
                n_val += len(target)
        val_loss /= max(n_val, 1)

        print(f"  [seed {seed}] Epoch {epoch+1:3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_val_loss': best_val_loss, 'patience_counter': patience_counter,
                    'best_model_state': best_state}, checkpoint_path)

        if patience_counter >= PATIENCE:
            print(f"  [seed {seed}] Early stopping at epoch {epoch+1}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_targets, all_lakes = [], [], []
    with torch.no_grad():
        for seq, static_vals, lake_idx, target, weight in test_loader:
            seq, static_vals = seq.to(device), static_vals.to(device)
            lake_idx_dev = lake_idx.to(device)
            pred = model(seq, static_vals, lake_idx_dev)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())
            all_lakes.append(lake_idx.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_lakes = np.concatenate(all_lakes)
    idx_to_lake = {v: k for k, v in data['lake_to_idx'].items()}
    lake_labels = [idx_to_lake[i] for i in all_lakes]

    denorm_preds, denorm_targets = [], []
    for pred, targ, lake in zip(all_preds, all_targets, lake_labels):
        vmin, vmax, degenerate = data['norm_stats'][lake][TARGET_ANOMALY]
        if degenerate:
            denorm_preds.append(vmin)
            denorm_targets.append(vmin)
        else:
            denorm_preds.append(pred * (vmax - vmin) + vmin)
            denorm_targets.append(targ * (vmax - vmin) + vmin)
    denorm_preds = np.array(denorm_preds)
    denorm_targets = np.array(denorm_targets)

    rmse = np.sqrt(mean_squared_error(denorm_targets, denorm_preds))
    mae  = mean_absolute_error(denorm_targets, denorm_preds)
    r2   = r2_score(denorm_targets, denorm_preds)
    nse_val = nse(denorm_targets, denorm_preds)
    print(f"  [seed {seed}] Test RMSE: {rmse:.4f} | R2: {r2:.4f}", flush=True)

    lake_df = pd.DataFrame({'lake': lake_labels, 'y_true': denorm_targets, 'y_pred': denorm_preds})
    raw_out_path = os.path.join(RESULTS_DIR, f'raw_preds_{version_name}_{VARIABLE}_{TARGET_HORIZON}_seed{seed}.pkl')
    with open(raw_out_path, 'wb') as rf:
        pickle.dump({'lake': np.array(lake_labels), 'y_true': denorm_targets, 'y_pred': denorm_preds}, rf)
    print(f'  [seed {seed}] Raw predictions saved to {raw_out_path}', flush=True)
    per_lake = []
    for lk, g in lake_df.groupby('lake'):
        ss_res = ((g.y_true - g.y_pred) ** 2).sum()
        ss_tot = ((g.y_true - g.y_true.mean()) ** 2).sum()
        per_lake.append({'lake': lk, 'n': len(g),
                         'r2_raw': 1 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
                         'true_range': g.y_true.max() - g.y_true.min(),
                         'ss_res': ss_res})
    per_lake_df = pd.DataFrame(per_lake).sort_values('n', ascending=False)
    per_lake_df['pct_of_total_ss_res'] = per_lake_df['ss_res'] / per_lake_df['ss_res'].sum() * 100
    out_path = os.path.join(RESULTS_DIR, f'per_lake_{version_name}_{VARIABLE}_{TARGET_HORIZON}_seed{seed}.csv')
    per_lake_df.to_csv(out_path, index=False)
    print(f"  [seed {seed}] Per-lake breakdown saved to {out_path}", flush=True)

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'nse': nse_val}

def run_experiment(version_name, dynamic_cols):
    print(f"\n{'='*60}\nEXPERIMENT: {version_name} (Fold {FOLD_ID})\n{'='*60}", flush=True)
    data = build_dataset(version_name, dynamic_cols)
    print(f"Train: {len(data['splits']['train']):,} | "
          f"Val: {len(data['splits']['val']):,} | "
          f"Test: {len(data['splits']['test']):,}", flush=True)

    seed_results = []
    for seed in SEEDS:
        r = run_single_seed(version_name, dynamic_cols, data, seed)
        seed_results.append(r)

    rmses = [r['rmse'] for r in seed_results]
    r2s   = [r['r2'] for r in seed_results]
    print(f"\n{version_name} summary across {len(SEEDS)} seeds:", flush=True)
    print(f"  RMSE: mean={np.mean(rmses):.4f} std={np.std(rmses):.4f}", flush=True)
    return {'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
            'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s), 'seed_results': seed_results}

DYNAMIC_COLS = DYNAMIC_COLS_BASE
with_fire = run_experiment('with_fire', DYNAMIC_COLS_BASE + FIRE_COLS)
no_fire   = run_experiment('no_fire', DYNAMIC_COLS_BASE)

print(f"\n{'='*60}\nFOLD {FOLD_ID} SUMMARY ({VARIABLE}, {TARGET_HORIZON})\n{'='*60}", flush=True)
print(f"With fire: RMSE={with_fire['rmse_mean']:.4f} +/- {with_fire['rmse_std']:.4f}", flush=True)
print(f"No fire:   RMSE={no_fire['rmse_mean']:.4f} +/- {no_fire['rmse_std']:.4f}", flush=True)
print(f"Direction: {'fire hurts' if with_fire['rmse_mean'] > no_fire['rmse_mean'] else 'fire helps'}", flush=True)
