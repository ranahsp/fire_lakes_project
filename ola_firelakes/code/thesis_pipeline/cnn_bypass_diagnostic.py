import sys
sys.path.insert(0, '/u/home/elwasila/firelakes/models/scripts')
import cnn_lstm_ablation as m
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

print("=== CNN-BYPASS DIAGNOSTIC ===", flush=True)
print("Testing whether scalar+static+embedding alone (no CNN/spatial) can learn", flush=True)

class BypassModel(nn.Module):
    """Same as CNNLSTM but skips the CNN entirely — scalar_seq goes
    straight into the LSTM, spatial_seq is ignored."""
    def __init__(self, n_scalar, n_static, n_lakes,
                 lstm_hidden=96, embed_dim=8, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_scalar, hidden_size=lstm_hidden,
                             num_layers=2, batch_first=True, dropout=dropout)
        self.lake_embed = nn.Embedding(n_lakes, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + embed_dim + n_static, 64),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, spatial_seq, scalar_seq, static_vals, lake_idx):
        lstm_out, (h_n, c_n) = self.lstm(scalar_seq)
        last_hidden = h_n[-1]
        lake_emb = self.embed_norm(self.lake_embed(lake_idx))
        combined = torch.cat([last_hidden, lake_emb, static_vals], dim=1)
        return self.fc(combined).squeeze(-1)

index = m.get_index('with_fire')
n_scalar = index['n_scalar']
n_static = 7
n_lakes = index['n_lakes']
idx_to_lake = {v: k for k, v in index['lake_to_idx'].items()}
norm_stats = index['norm_stats']

torch.manual_seed(42)
np.random.seed(42)

train_loader = m.PatchFileShuffledLoader('with_fire', m.VARIABLE, m.TARGET_HORIZON, 'train', 256, shuffle=True)
val_loader   = m.PatchFileShuffledLoader('with_fire', m.VARIABLE, m.TARGET_HORIZON, 'val', 256, shuffle=False)
test_loader  = m.PatchFileShuffledLoader('with_fire', m.VARIABLE, m.TARGET_HORIZON, 'test', 256, shuffle=False)

model = BypassModel(n_scalar, n_static, n_lakes).to(m.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

for epoch in range(10):
    model.train()
    train_loss, n_train = 0.0, 0
    for spatial_seq, scalar_seq, static_vals, lake_idx, target, weight in train_loader:
        scalar_seq, static_vals = scalar_seq.to(m.device), static_vals.to(m.device)
        lake_idx, target, weight = lake_idx.to(m.device), target.to(m.device), weight.to(m.device)
        optimizer.zero_grad()
        pred = model(None, scalar_seq, static_vals, lake_idx)
        loss = m.weighted_mse(pred, target, weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(target)
        n_train += len(target)
    train_loss /= n_train

    model.eval()
    val_loss, n_val = 0.0, 0
    with torch.no_grad():
        for spatial_seq, scalar_seq, static_vals, lake_idx, target, weight in val_loader:
            scalar_seq, static_vals = scalar_seq.to(m.device), static_vals.to(m.device)
            lake_idx, target, weight = lake_idx.to(m.device), target.to(m.device), weight.to(m.device)
            pred = model(None, scalar_seq, static_vals, lake_idx)
            loss = m.weighted_mse(pred, target, weight)
            val_loss += loss.item() * len(target)
            n_val += len(target)
    val_loss /= n_val
    print(f"Epoch {epoch+1} | Train: {train_loss:.5f} | Val: {val_loss:.5f}", flush=True)

model.eval()
all_preds, all_targets, all_lakes = [], [], []
with torch.no_grad():
    for spatial_seq, scalar_seq, static_vals, lake_idx, target, weight in test_loader:
        scalar_seq, static_vals = scalar_seq.to(m.device), static_vals.to(m.device)
        lake_idx_dev = lake_idx.to(m.device)
        pred = model(None, scalar_seq, static_vals, lake_idx_dev)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.numpy())
        all_lakes.append(lake_idx.numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)
all_lakes = np.concatenate(all_lakes)
lake_labels = [idx_to_lake[i] for i in all_lakes]

denorm_preds, denorm_targets = [], []
for pred, targ, lake in zip(all_preds, all_targets, lake_labels):
    vmin, vmax, degenerate = norm_stats[lake][m.TARGET_ANOMALY]
    if degenerate:
        denorm_preds.append(vmin); denorm_targets.append(vmin)
    else:
        denorm_preds.append(pred * (vmax - vmin) + vmin)
        denorm_targets.append(targ * (vmax - vmin) + vmin)

rmse = np.sqrt(mean_squared_error(denorm_targets, denorm_preds))
r2 = r2_score(denorm_targets, denorm_preds)
print(f"\nBYPASS TEST RESULT — RMSE: {rmse:.4f} | R2: {r2:.4f}", flush=True)
print("=== DIAGNOSTIC COMPLETE ===", flush=True)
