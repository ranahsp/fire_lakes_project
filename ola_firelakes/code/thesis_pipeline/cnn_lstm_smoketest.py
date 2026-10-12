import torch
import pickle
import os
import numpy as np

PATCH_CACHE_DIR = "/u/home/elwasila/firelakes/data/model_ready/patch_cache"

print("=== SMOKE TEST: loading one lake file and running one forward pass ===")

# Load the index to find a small lake's file
index_path = os.path.join(PATCH_CACHE_DIR, 'with_fire_chla_1week_INDEX.pkl')
with open(index_path, 'rb') as f:
    index = pickle.load(f)

print(f"n_spatial: {index['n_spatial']} | n_scalar: {index['n_scalar']} | n_lakes: {index['n_lakes']}")

# Find yardi's file (smallest lake) in train split
train_files = index['lake_files']['train']
yardi_file = [f for f in train_files if 'yardi' in f][0]
print(f"Loading small test file: {yardi_file}")

with open(os.path.join(PATCH_CACHE_DIR, yardi_file), 'rb') as f:
    samples = pickle.load(f)

print(f"Samples in file: {len(samples)}")
spatial_patch, scalar_seq, static_vals, lake_idx, target, weight = samples[0]
print(f"spatial_patch shape: {spatial_patch.shape} dtype: {spatial_patch.dtype}")
print(f"scalar_seq shape: {scalar_seq.shape}")
print(f"static_vals shape: {static_vals.shape}")
print(f"lake_idx: {lake_idx} | target: {target} | weight: {weight}")

# Now test the actual model architecture with a tiny batch
import sys
sys.path.insert(0, '/u/home/elwasila/firelakes/models/scripts')
from cnn_lstm_ablation import CNNLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model = CNNLSTM(
    n_spatial=index['n_spatial'],
    n_scalar=index['n_scalar'],
    n_static=7,
    n_lakes=index['n_lakes'],
).to(device)

batch_size = 4
spatial_batch = torch.tensor(np.stack([s[0].astype(np.float32) for s in samples[:batch_size]])).to(device)
scalar_batch = torch.tensor(np.stack([s[1].astype(np.float32) for s in samples[:batch_size]])).to(device)
static_batch = torch.tensor(np.stack([s[2].astype(np.float32) for s in samples[:batch_size]])).to(device)
lake_idx_batch = torch.tensor([s[3] for s in samples[:batch_size]], dtype=torch.long).to(device)

print(f"spatial_batch shape: {spatial_batch.shape}")
print(f"scalar_batch shape: {scalar_batch.shape}")

output = model(spatial_batch, scalar_batch, static_batch, lake_idx_batch)
print(f"\nModel output shape: {output.shape}")
print(f"Model output values: {output}")

print("\n=== SMOKE TEST PASSED — model runs end-to-end without error ===")
