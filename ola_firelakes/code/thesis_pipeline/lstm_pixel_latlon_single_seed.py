"""
Trains a SINGLE specified seed for the pixel-level LSTM + lat/lon
model, reusing the already-cached sequence data built by
lstm_pixel_latlon.py. Used to complete a combination where some
seeds already finished (e.g. after a walltime kill), without
re-running already-completed seeds.

This is a near-exact copy of lstm_pixel_latlon.py's training logic,
just restricted to one seed passed on the command line, and it loads
the EXISTING cache rather than rebuilding it.

Usage: python3 lstm_pixel_latlon_single_seed.py <variable> <horizon> <version> <seed>
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
RESULTS_DIR    = "/u/home/elwasila/firelakes/models/results/LSTM_pixel_latlon"
CHECKPOINT_DIR = "/u/home/elwasila/firelakes/models/checkpoints/pixel_latlon"
SEQ_CACHE_DIR  = "/u/home/elwasila/firelakes/data/model_ready/seq_cache/pixel_latlon"

VARIABLE = sys.argv[1]
TARGET_HORIZON = sys.argv[2]
VERSION = sys.argv[3]
SEED = int(sys.argv[4])
assert VARIABLE in ['chla', 'tsm']
assert VERSION in ['with_fire', 'no_fire']

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Single-seed run | Variable: {VARIABLE} | Horizon: {TARGET_HORIZON} | Version: {VERSION} | Seed: {SEED}", flush=True)
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
DYNAMIC_COLS = DYNAMIC_COLS_BASE + (FIRE_COLS if VERSION == 'with_fire' else [])
STATIC_COLS = ['depth_avg_m', 'vol_total_km3', 'wshd_area_km2',
               'res_time_days', 'elevation_m', 'lc_tre_usu', 'slp_dg_uav',
               'rel_lat', 'rel_lon']

cache_path = os.path.join(SEQ_CACHE_DIR, f'{VERSION}_{VARIABLE}_{TARGET_HORIZON}_L{LOOKBACK}.pkl')
print(f"Loading cached sequences: {cache_path}", flush=True)
with open(cache_path, 'rb') as f:
    data = pickle.load(f)

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

torch.manual_seed(SEED)
np.random.seed(SEED)

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

    if epoch % 5 == 0:
        print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f}", flush=True)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = model.state_dict()
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        print(f"  Early stopping at epoch {epoch+1}", flush=True)
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
print(f"\nSeed {SEED} Test RMSE: {rmse:.4f} | R2: {r2:.4f} | "
      f"pred_std/obs_std: {ratio:.4f} | correlation: {corr:.4f}", flush=True)

raw_out_path = os.path.join(RESULTS_DIR, f'raw_preds_{VERSION}_{VARIABLE}_{TARGET_HORIZON}_seed{SEED}.pkl')
with open(raw_out_path, 'wb') as rf:
    pickle.dump({'lake': np.array(lake_labels), 'y_true': denorm_targets, 'y_pred': denorm_preds}, rf)
print(f"Saved: {raw_out_path}", flush=True)
