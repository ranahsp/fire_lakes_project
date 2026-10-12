import pickle
import os

PATCH_CACHE_DIR = "/u/home/elwasila/firelakes/data/model_ready/patch_cache"

configs = [
    ('with_fire', 'chla', '1week'),
    ('no_fire', 'chla', '1week'),
    ('with_fire', 'tsm', '1week'),
    ('no_fire', 'tsm', '1week'),
]

for version, variable, horizon in configs:
    index_path = os.path.join(PATCH_CACHE_DIR, f'{version}_{variable}_{horizon}_INDEX.pkl')
    if not os.path.exists(index_path):
        print(f"SKIP (not found): {index_path}", flush=True)
        continue

    with open(index_path, 'rb') as f:
        index = pickle.load(f)

    if 'lake_file_counts' in index:
        print(f"Already has counts, skipping: {index_path}", flush=True)
        continue

    print(f"Processing: {version}/{variable}/{horizon}", flush=True)
    lake_file_counts = {'train': {}, 'val': {}, 'test': {}}

    for split in ['train', 'val', 'test']:
        for fname in index['lake_files'][split]:
            fpath = os.path.join(PATCH_CACHE_DIR, fname)
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            lake_file_counts[split][fname] = len(data)
            print(f"  {fname}: {len(data)} samples", flush=True)

    index['lake_file_counts'] = lake_file_counts
    with open(index_path, 'wb') as f:
        pickle.dump(index, f)
    print(f"  Updated and saved: {index_path}", flush=True)

print("\nAll configs processed.", flush=True)
