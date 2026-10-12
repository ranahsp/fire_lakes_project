"""
Lake-aggregated LSTM: tests Gloria's hypothesis directly.

Instead of training on individual pixels (where catchment-level fire/
weather predictors are identical across pixels but the target varies
per pixel), this aggregates EVERYTHING to one value per lake per week
-- median across all pixels, for both the target and the per-pixel
water-quality inputs (chla_anomaly, tsm_anomaly, lswt_anomaly), which
are the only inputs that currently vary within a lake. Fire/weather/
static predictors are also aggregated (median) for pipeline
consistency, though we already confirmed these are effectively
uniform across pixels already.

If this lake-level model shows genuine week-to-week responsiveness
(unlike the flat, near-constant per-pixel predictions found earlier),
that would support Gloria's hypothesis: the per-pixel resolution
mismatch, not a fundamental lack of fire signal, is a major driver of
the flat-prediction problem. If it's still flat even at this coarser
scale, the problem likely lies elsewhere (e.g. genuinely weak fire
signal, or the lake-identity shortcut itself).

Training unit is now (lake, week), not (lake, cell_id, week) -- with
~29 lakes and roughly 1,000 weeks each, this is a MUCH smaller
dataset than the pixel-level version, so training should be
correspondingly much faster.

Usage: python3 lstm_lake_aggregated.py <variable> <horizon>
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
RESULTS_DIR    = "/u/home/elwasila/firelakes/models/results/LSTM_lake_aggregated"
CHECKPOINT_DIR = "/u/home/elwasila/firelakes/models/checkpoints/lake_aggregated"
CACHE_DIR      = "/u/home/elwasila/firelakes/data/model_ready/lake_aggregated_cache"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

VARIABLE = sys.argv[1] if len(sys.argv) > 1 else 'chla'
TARGET_HORIZON = sys.argv[2] if len(sys.argv) > 2 else '1week'
VERSION = sys.argv[3] if len(sys.argv) > 3 else 'with_fire'
assert VARIABLE in ['chla', 'tsm']
assert VERSION in ['with_fire', 'no_fire']

TARGET_BASE = f'{VARIABLE}_next_{TARGET_HORIZON}'
TARGET_ANOMALY = f'{TARGET_BASE}_anomaly'

LOOKBACK    = 10
BATCH_SIZE  = 32   # much smaller dataset -- smaller batch is appropriate
HIDDEN_SIZE = 64   # smaller model -- far fewer samples than the pixel-level version
EMBED_DIM   = 16
DROPOUT     = 0.3
LR          = 0.0005
MAX_EPOCHS  = 100
PATIENCE    = 10
GRAD_CLIP   = 1.0
NEUTRAL_VALUE = 0.5
SEEDS       = [42, 123, 2024]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Lake-aggregated LSTM | Variable: {VARIABLE} | Horizon: {TARGET_HORIZON} | Version: {VERSION}", flush=True)
print(f"Using device: {device}", flush=True)

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
DYNAMIC_COLS = DYNAMIC_COLS_BASE + (FIRE_COLS if VERSION == 'with_fire' else [])

def safe_minmax(arr):
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (0.0, 1.0, True)
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return (vmin, vmax, True)
    return (vmin, vmax, False)

def build_lake_aggregated_dataset():
    cache_path = os.path.join(CACHE_DIR, f'{VERSION}_{VARIABLE}_{TARGET_HORIZON}_L{LOOKBACK}.pkl')
    if os.path.exists(cache_path):
        print(f"Loading cached lake-aggregated sequences: {cache_path}", flush=True)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("Building lake-aggregated sequences...", flush=True)
    train_df = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_train.parquet'))
    val_df   = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_val.parquet'))
    test_df  = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_test.parquet'))
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    del train_df, val_df, test_df

    with open(os.path.join(DATA_DIR, f'{VARIABLE}_lake_weights_v2.json')) as f:
        lake_weights = json.load(f)

    lake_names = sorted(full_df['lake_name'].unique())
    lake_to_idx = {name: i for i, name in enumerate(lake_names)}

    # KEY STEP: aggregate to ONE ROW PER (lake, year, week) -- median
    # across all pixels, for every dynamic predictor AND the target.
    # This directly removes the per-pixel/catchment-level resolution
    # mismatch Gloria identified: every input the model sees is now
    # genuinely at the same (lake-week) resolution as the target.
    agg_cols = DYNAMIC_COLS + [TARGET_ANOMALY]
    lake_weekly = full_df.groupby(['lake_name', 'year', 'week'])[agg_cols].median().reset_index()

    # Static columns are already one value per lake -- just take the first
    static_by_lake = full_df.groupby('lake_name')[STATIC_COLS].first().reset_index()
    lake_weekly = lake_weekly.merge(static_by_lake, on='lake_name', how='left')

    print(f"Aggregated to {len(lake_weekly):,} (lake, week) rows "
          f"from {len(full_df):,} original pixel-week rows", flush=True)
    del full_df

    ALL_NORM_COLS = DYNAMIC_COLS + STATIC_COLS + [TARGET_ANOMALY]
    norm_stats = {}
    train_mask = (lake_weekly['year'] >= 2002) & (lake_weekly['year'] <= 2018)
    for lake in lake_names:
        lt = lake_weekly[train_mask & (lake_weekly['lake_name'] == lake)]
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

    for lake, group in lake_weekly.groupby('lake_name'):
        if lake not in norm_stats:
            continue
        group = group.sort_values(['year', 'week']).reset_index(drop=True)

        feat_matrix = np.zeros((len(group), len(DYNAMIC_COLS)), dtype=np.float32)
        for j, col in enumerate(DYNAMIC_COLS):
            feat_matrix[:, j] = normalize(group[col].values, lake, col).astype(np.float32)
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)

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

    result = {'splits': split_samples, 'lake_to_idx': lake_to_idx,
              'norm_stats': norm_stats, 'n_lakes': len(lake_names)}
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Cached to: {cache_path}", flush=True)
    for split_name, samples in split_samples.items():
        print(f"  {split_name}: {len(samples)} samples", flush=True)
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
    def __init__(self, n_dynamic, n_static, n_lakes, hidden_size=64, embed_dim=16, dropout=0.3):
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

def run_single_seed(data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = SeqDataset(data['splits']['train'])
    val_ds   = SeqDataset(data['splits']['val'])
    test_ds  = SeqDataset(data['splits']['test'])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FireLakeLSTM(len(DYNAMIC_COLS), len(STATIC_COLS), data['n_lakes'],
                          HIDDEN_SIZE, EMBED_DIM, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    best_val_loss, patience_counter = float('inf'), 0
    best_state = model.state_dict()

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss, n_train = 0.0, 0
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
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for seq, static_vals, lake_idx, target, weight in val_loader:
                seq, static_vals = seq.to(device), static_vals.to(device)
                lake_idx, target, weight = lake_idx.to(device), target.to(device), weight.to(device)
                pred = model(seq, static_vals, lake_idx)
                loss = weighted_mse(pred, target, weight)
                val_loss += loss.item() * len(target)
                n_val += len(target)
        val_loss /= max(n_val, 1)

        if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
            print(f"  [seed {seed}] Epoch {epoch+1:3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
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
            denorm_preds.append(vmin); denorm_targets.append(vmin)
        else:
            denorm_preds.append(pred * (vmax - vmin) + vmin)
            denorm_targets.append(targ * (vmax - vmin) + vmin)
    denorm_preds = np.array(denorm_preds)
    denorm_targets = np.array(denorm_targets)

    rmse = np.sqrt(mean_squared_error(denorm_targets, denorm_preds))
    r2 = r2_score(denorm_targets, denorm_preds)
    pred_std, obs_std = denorm_preds.std(), denorm_targets.std()
    ratio = pred_std / obs_std if obs_std > 0 else float('nan')
    corr = np.corrcoef(denorm_targets, denorm_preds)[0, 1] if obs_std > 0 and pred_std > 0 else float('nan')
    print(f"  [seed {seed}] Test RMSE: {rmse:.4f} | R2: {r2:.4f} | "
          f"pred_std/obs_std: {ratio:.4f} | correlation: {corr:.4f}", flush=True)

    raw_out_path = os.path.join(RESULTS_DIR, f'raw_preds_{VERSION}_{VARIABLE}_{TARGET_HORIZON}_seed{seed}.pkl')
    with open(raw_out_path, 'wb') as rf:
        pickle.dump({'lake': np.array(lake_labels), 'y_true': denorm_targets, 'y_pred': denorm_preds}, rf)

    return {'rmse': rmse, 'r2': r2, 'pred_std': pred_std, 'obs_std': obs_std, 'ratio': ratio, 'corr': corr}

data = build_lake_aggregated_dataset()
seed_results = [run_single_seed(data, seed) for seed in SEEDS]

rmses = [r['rmse'] for r in seed_results]
ratios = [r['ratio'] for r in seed_results]
corrs = [r['corr'] for r in seed_results]
print(f"\n{'='*60}\nSUMMARY ({VARIABLE}, {TARGET_HORIZON}, {VERSION})\n{'='*60}", flush=True)
print(f"RMSE: mean={np.mean(rmses):.4f} std={np.std(rmses):.4f}", flush=True)
print(f"pred_std/obs_std ratio: mean={np.mean(ratios):.4f}", flush=True)
print(f"Correlation: mean={np.mean(corrs):.4f}", flush=True)
