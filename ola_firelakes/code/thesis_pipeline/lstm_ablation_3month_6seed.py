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

DATA_DIR      = "/u/home/elwasila/firelakes/data/model_ready"
RESULTS_DIR   = "/u/home/elwasila/firelakes/models/results/LSTM"
CHECKPOINT_DIR = "/u/home/elwasila/firelakes/models/checkpoints"
SEQ_CACHE_DIR = "/u/home/elwasila/firelakes/data/model_ready/seq_cache"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SEQ_CACHE_DIR, exist_ok=True)

VARIABLE = sys.argv[1] if len(sys.argv) > 1 else 'chla'
TARGET_HORIZON = sys.argv[2] if len(sys.argv) > 2 else '1week'
assert VARIABLE in ['chla', 'tsm']
assert TARGET_HORIZON in ['1week', '1month', '3month', '6month', '1year', '2year']

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
SEEDS       = [42, 123, 2024, 7, 99, 777]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Variable: {VARIABLE} | Horizon: {TARGET_HORIZON}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
        print(f"Loading cached sequences: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Building sequences for {version_name}...")
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
    train_mask = (full_df['year'] >= 2002) & (full_df['year'] <= 2018)
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

    splits = {'train': (2002, 2018), 'val': (2019, 2020), 'test': (2021, 2022)}
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
    print(f"Cached to: {cache_path}")
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
        print(f"  [seed {seed}] Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        train_loss = 0.0
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
        train_loss /= max(len(train_ds), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, static_vals, lake_idx, target, weight in val_loader:
                seq, static_vals = seq.to(device), static_vals.to(device)
                lake_idx, target, weight = lake_idx.to(device), target.to(device), weight.to(device)
                pred = model(seq, static_vals, lake_idx)
                loss = weighted_mse(pred, target, weight)
                val_loss += loss.item() * len(target)
        val_loss /= max(len(val_ds), 1)

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
    print(f"  [seed {seed}] Test RMSE: {rmse:.4f} | R2: {r2:.4f}")

    lake_df = pd.DataFrame({'lake': lake_labels, 'y_true': denorm_targets, 'y_pred': denorm_preds})
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
    print(f"  [seed {seed}] Per-lake breakdown saved to {out_path}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'nse': nse_val}

def run_experiment(version_name, dynamic_cols):
    print(f"\n{'='*60}\nEXPERIMENT: {version_name}\n{'='*60}")
    data = build_dataset(version_name, dynamic_cols)
    print(f"Train: {len(data['splits']['train']):,} | "
          f"Val: {len(data['splits']['val']):,} | "
          f"Test: {len(data['splits']['test']):,}")

    seed_results = []
    for seed in SEEDS:
        r = run_single_seed(version_name, dynamic_cols, data, seed)
        seed_results.append(r)

    rmses = [r['rmse'] for r in seed_results]
    r2s   = [r['r2'] for r in seed_results]
    print(f"\n{version_name} summary across {len(SEEDS)} seeds:")
    print(f"  RMSE: mean={np.mean(rmses):.4f} std={np.std(rmses):.4f}")
    print(f"  R2:   mean={np.mean(r2s):.4f} std={np.std(r2s):.4f}")

    return {
        'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
        'mae_mean': np.mean([r['mae'] for r in seed_results]),
        'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
        'nse_mean': np.mean([r['nse'] for r in seed_results]),
        'seed_results': seed_results,
    }

with_fire = run_experiment('with_fire', DYNAMIC_COLS_BASE + FIRE_COLS)
no_fire   = run_experiment('no_fire', DYNAMIC_COLS_BASE)

comparison = pd.DataFrame([
    {'Model': 'LSTM', 'Horizon': TARGET_HORIZON, 'Version': 'With fire',
     'RMSE_mean': with_fire['rmse_mean'], 'RMSE_std': with_fire['rmse_std'],
     'R2_mean': with_fire['r2_mean'], 'R2_std': with_fire['r2_std']},
    {'Model': 'LSTM', 'Horizon': TARGET_HORIZON, 'Version': 'Without fire',
     'RMSE_mean': no_fire['rmse_mean'], 'RMSE_std': no_fire['rmse_std'],
     'R2_mean': no_fire['r2_mean'], 'R2_std': no_fire['r2_std']},
])
print(f"\n{'='*60}\nMULTI-SEED COMPARISON TABLE ({VARIABLE}, {TARGET_HORIZON})\n{'='*60}")
print(comparison.to_string(index=False))

n_with_wins = sum(1 for wf_r2, nf_r2 in
                  zip([r['r2'] for r in with_fire['seed_results']],
                      [r['r2'] for r in no_fire['seed_results']])
                  if wf_r2 > nf_r2)
print(f"\nOut of {len(SEEDS)} seeds, 'With Fire' had higher R2 in {n_with_wins} of them.")

comparison.to_csv(os.path.join(RESULTS_DIR, f'lstm_multiseed_comparison_{VARIABLE}_{TARGET_HORIZON}.csv'), index=False)
print(f"\nLSTM multi-seed ablation complete ({VARIABLE}, {TARGET_HORIZON}). Results saved to {RESULTS_DIR}")
