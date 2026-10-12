import pandas as pd
import numpy as np
import json
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR      = "/u/home/elwasila/firelakes/data/model_ready"
RESULTS_DIR   = "/u/home/elwasila/firelakes/models/results/LSTM"
CHECKPOINT_DIR = "/u/home/elwasila/firelakes/models/checkpoints"
SEQ_CACHE_DIR = "/u/home/elwasila/firelakes/data/model_ready/seq_cache"

VARIABLE = 'chla'
TARGET_HORIZON = '3month'
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
EXTRA_SEEDS = [7, 99, 777]  # new seeds, in addition to [42, 123, 2024]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"EXTRA SEEDS test | Variable: {VARIABLE} | Horizon: {TARGET_HORIZON}")
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

def build_dataset(version_name, dynamic_cols):
    # Reuses the SAME cache already built during the original 3-month run
    cache_path = os.path.join(SEQ_CACHE_DIR, f'{version_name}_{VARIABLE}_{TARGET_HORIZON}_L{LOOKBACK}.pkl')
    print(f"Loading cached sequences: {cache_path}")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

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
    r2   = r2_score(denorm_targets, denorm_preds)
    print(f"  [seed {seed}] Test RMSE: {rmse:.4f} | R2: {r2:.4f}")
    return {'rmse': rmse, 'r2': r2}

for version_name, cols in [('with_fire', DYNAMIC_COLS_BASE + FIRE_COLS),
                             ('no_fire', DYNAMIC_COLS_BASE)]:
    print(f"\n{'='*60}\nEXPERIMENT: {version_name} (extra seeds)\n{'='*60}")
    data = build_dataset(version_name, cols)
    results = []
    for seed in EXTRA_SEEDS:
        r = run_single_seed(version_name, cols, data, seed)
        results.append(r)
    rmses = [r['rmse'] for r in results]
    r2s = [r['r2'] for r in results]
    print(f"\n{version_name} EXTRA seeds summary: RMSE mean={np.mean(rmses):.4f} std={np.std(rmses):.4f} | "
          f"R2 mean={np.mean(r2s):.4f} std={np.std(r2s):.4f}")
    print(f"All 6 seeds combined R2 values would need manual merge with original [42,123,2024] results.")

print("\nExtra-seed test complete for chla 3-month.")
