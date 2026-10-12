import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR       = "/u/home/elwasila/firelakes/data/model_ready"
PATCH_CACHE_DIR = "/u/home/elwasila/firelakes/data/model_ready/patch_cache"
RESULTS_DIR    = "/u/home/elwasila/firelakes/models/results/CNN_LSTM"
CHECKPOINT_DIR = "/u/home/elwasila/firelakes/models/checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

VARIABLE = sys.argv[1] if len(sys.argv) > 1 else 'chla'
TARGET_HORIZON = sys.argv[2] if len(sys.argv) > 2 else '1week'
VERSION_ARG = sys.argv[3] if len(sys.argv) > 3 else None

assert VARIABLE in ['chla', 'tsm']

TARGET_BASE = f'{VARIABLE}_next_{TARGET_HORIZON}'
TARGET_ANOMALY = f'{TARGET_BASE}_anomaly'

LOOKBACK    = 10
BATCH_SIZE  = 256
CNN_CHANNELS = 16
LSTM_HIDDEN = 96
EMBED_DIM   = 8
DROPOUT     = 0.3
LR          = 0.00005
MAX_EPOCHS  = 15
PATIENCE    = 3
GRAD_CLIP   = 1.0
WEIGHT_DECAY = 1e-4
SEEDS       = [42, 123, 2024]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CNN-LSTM | Variable: {VARIABLE} | Horizon: {TARGET_HORIZON}", flush=True)
print(f"Using device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

class PatchFileShuffledLoader:
    def __init__(self, version, variable, horizon, split, batch_size, shuffle=True):
        index_path = os.path.join(PATCH_CACHE_DIR, f'{version}_{variable}_{horizon}_INDEX.pkl')
        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)
        self.files = list(self.index['lake_files'][split])
        self.patch_cache_dir = PATCH_CACHE_DIR
        self.batch_size = batch_size
        self.shuffle = shuffle

        counts_lookup = self.index.get('lake_file_counts', {}).get(split, {})
        self.total_samples = sum(counts_lookup.get(f, 0) for f in self.files)
        print(f"  {split}: {len(self.files)} lake files, {self.total_samples:,} samples", flush=True)

    def __len__(self):
        return (self.total_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        files = list(self.files)
        if self.shuffle:
            np.random.shuffle(files)

        buffer = []
        for fname in files:
            fpath = os.path.join(self.patch_cache_dir, fname)
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            if self.shuffle:
                idxs = np.random.permutation(len(data))
            else:
                idxs = range(len(data))

            for i in idxs:
                buffer.append(data[i])
                if len(buffer) == self.batch_size:
                    yield self._collate(buffer)
                    buffer = []
            del data
        if len(buffer) > 0:
            yield self._collate(buffer)

    def _collate(self, samples):
        spatial = torch.tensor(np.stack([s[0].astype(np.float32) for s in samples]))
        scalar  = torch.tensor(np.stack([s[1].astype(np.float32) for s in samples]))
        static  = torch.tensor(np.stack([s[2].astype(np.float32) for s in samples]))
        lake_idx = torch.tensor([s[3] for s in samples], dtype=torch.long)
        target  = torch.tensor([float(s[4]) for s in samples], dtype=torch.float32)
        weight  = torch.tensor([float(s[5]) for s in samples], dtype=torch.float32)
        return spatial, scalar, static, lake_idx, target, weight

def get_index(version):
    index_path = os.path.join(PATCH_CACHE_DIR, f'{version}_{VARIABLE}_{TARGET_HORIZON}_INDEX.pkl')
    with open(index_path, 'rb') as f:
        return pickle.load(f)

class SpatialCNNEncoder(nn.Module):
    def __init__(self, n_spatial_channels, out_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_spatial_channels, CNN_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(CNN_CHANNELS, out_dim)
        self.norm = nn.LayerNorm(out_dim)  # stabilizes CNN output scale before LSTM

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x).flatten(1)
        x = self.fc(x)
        return self.norm(x)

class CNNLSTM(nn.Module):
    def __init__(self, n_spatial, n_scalar, n_static, n_lakes,
                 cnn_out_dim=16, lstm_hidden=96, embed_dim=16, dropout=0.3):
        super().__init__()
        self.cnn_encoder = SpatialCNNEncoder(n_spatial, out_dim=cnn_out_dim)
        self.lstm = nn.LSTM(input_size=cnn_out_dim + n_scalar, hidden_size=lstm_hidden,
                             num_layers=2, batch_first=True, dropout=dropout)
        self.lake_embed = nn.Embedding(n_lakes, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + embed_dim + n_static, 64),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, spatial_seq, scalar_seq, static_vals, lake_idx):
        B, T = spatial_seq.shape[0], spatial_seq.shape[1]
        spatial_flat = spatial_seq.reshape(B * T, *spatial_seq.shape[2:])
        cnn_feats = self.cnn_encoder(spatial_flat)
        cnn_feats = cnn_feats.reshape(B, T, -1)

        combined_seq = torch.cat([cnn_feats, scalar_seq], dim=2)
        lstm_out, (h_n, c_n) = self.lstm(combined_seq)
        last_hidden = h_n[-1]

        lake_emb = self.embed_norm(self.lake_embed(lake_idx))
        combined = torch.cat([last_hidden, lake_emb, static_vals], dim=1)
        return self.fc(combined).squeeze(-1)

def weighted_mse(pred, target, weight):
    return ((pred - target)**2 * weight).mean()

def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)

def run_single_seed(version_name, seed, n_spatial, n_scalar, n_static, n_lakes,
                     norm_stats, idx_to_lake):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = PatchFileShuffledLoader(version_name, VARIABLE, TARGET_HORIZON, 'train', BATCH_SIZE, shuffle=True)
    val_loader   = PatchFileShuffledLoader(version_name, VARIABLE, TARGET_HORIZON, 'val', BATCH_SIZE, shuffle=False)
    test_loader  = PatchFileShuffledLoader(version_name, VARIABLE, TARGET_HORIZON, 'test', BATCH_SIZE, shuffle=False)

    model = CNNLSTM(n_spatial, n_scalar, n_static, n_lakes,
                     cnn_out_dim=16, lstm_hidden=LSTM_HIDDEN, embed_dim=EMBED_DIM, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'cnnlstm_{version_name}_{VARIABLE}_{TARGET_HORIZON}_seed{seed}.pt')
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
        for spatial_seq, scalar_seq, static_vals, lake_idx, target, weight in train_loader:
            spatial_seq, scalar_seq = spatial_seq.to(device), scalar_seq.to(device)
            static_vals, lake_idx = static_vals.to(device), lake_idx.to(device)
            target, weight = target.to(device), weight.to(device)

            optimizer.zero_grad()
            pred = model(spatial_seq, scalar_seq, static_vals, lake_idx)
            loss = weighted_mse(pred, target, weight)
            loss.backward()

            # DIAGNOSTIC: gradient norm, printed once per epoch (first batch only)
            if n_train == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                pred_std = pred.detach().std().item()
                pred_min = pred.detach().min().item()
                pred_max = pred.detach().max().item()
                print(f"  [seed {seed}] Epoch {epoch+1} DIAG (first batch) — grad_norm: {total_norm:.5f} | "
                      f"pred_std: {pred_std:.5f} | pred_range: [{pred_min:.4f}, {pred_max:.4f}]", flush=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * len(target)
            n_train += len(target)
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for spatial_seq, scalar_seq, static_vals, lake_idx, target, weight in val_loader:
                spatial_seq, scalar_seq = spatial_seq.to(device), scalar_seq.to(device)
                static_vals, lake_idx = static_vals.to(device), lake_idx.to(device)
                target, weight = target.to(device), weight.to(device)
                pred = model(spatial_seq, scalar_seq, static_vals, lake_idx)
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
        for spatial_seq, scalar_seq, static_vals, lake_idx, target, weight in test_loader:
            spatial_seq, scalar_seq = spatial_seq.to(device), scalar_seq.to(device)
            static_vals, lake_idx_dev = static_vals.to(device), lake_idx.to(device)
            pred = model(spatial_seq, scalar_seq, static_vals, lake_idx_dev)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())
            all_lakes.append(lake_idx.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_lakes = np.concatenate(all_lakes)
    lake_labels = [idx_to_lake[i] for i in all_lakes]

    # DIAGNOSTIC: R2 in normalized space, before denormalization
    norm_r2 = r2_score(all_targets, all_preds)
    print(f"  [seed {seed}] DIAGNOSTIC — Normalized-space R2: {norm_r2:.4f}", flush=True)
    print(f"  [seed {seed}] DIAGNOSTIC — pred range: [{all_preds.min():.4f}, {all_preds.max():.4f}] | target range: [{all_targets.min():.4f}, {all_targets.max():.4f}]", flush=True)

    denorm_preds, denorm_targets = [], []
    for pred, targ, lake in zip(all_preds, all_targets, lake_labels):
        vmin, vmax, degenerate = norm_stats[lake][TARGET_ANOMALY]
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
    print(f"  [seed {seed}] Test RMSE: {rmse:.4f} | R2: {r2:.4f}", flush=True)

    # Per-lake breakdown -- check whether pooled R2 is dominated by a few large-range lakes
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
    print(per_lake_df.head(10).to_string(index=False), flush=True)

    return {'rmse': rmse, 'r2': r2}

def run_experiment(version_name):
    print(f"\n{'='*60}\nEXPERIMENT: {version_name}\n{'='*60}", flush=True)
    index = get_index(version_name)
    n_spatial = index['n_spatial']
    n_scalar = index['n_scalar']
    n_static = 7
    n_lakes = index['n_lakes']
    idx_to_lake = {v: k for k, v in index['lake_to_idx'].items()}
    norm_stats = index['norm_stats']

    seed_results = []
    for seed in SEEDS:
        r = run_single_seed(version_name, seed, n_spatial, n_scalar, n_static, n_lakes,
                             norm_stats, idx_to_lake)
        seed_results.append(r)

    rmses = [r['rmse'] for r in seed_results]
    r2s   = [r['r2'] for r in seed_results]
    print(f"\n{version_name} summary: RMSE mean={np.mean(rmses):.4f} std={np.std(rmses):.4f} | "
          f"R2 mean={np.mean(r2s):.4f} std={np.std(r2s):.4f}", flush=True)
    return {'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
            'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s), 'seed_results': seed_results}

if __name__ == "__main__":
    if VERSION_ARG:
        versions_to_run = [VERSION_ARG]
    else:
        versions_to_run = ['with_fire', 'no_fire']

    results = {}
    for v in versions_to_run:
        results[v] = run_experiment(v)

    if len(results) == 2:
        comparison = pd.DataFrame([
            {'Model': 'CNN-LSTM', 'Version': 'With fire', **{k: results['with_fire'][k] for k in ['rmse_mean','rmse_std','r2_mean','r2_std']}},
            {'Model': 'CNN-LSTM', 'Version': 'Without fire', **{k: results['no_fire'][k] for k in ['rmse_mean','rmse_std','r2_mean','r2_std']}},
        ])
        print(f"\n{'='*60}\nCNN-LSTM COMPARISON TABLE ({VARIABLE})\n{'='*60}", flush=True)
        print(comparison.to_string(index=False), flush=True)
        comparison.to_csv(os.path.join(RESULTS_DIR, f'cnnlstm_comparison_{VARIABLE}_{TARGET_HORIZON}.csv'), index=False)

    print(f"\nCNN-LSTM experiment complete ({VARIABLE}).", flush=True)
