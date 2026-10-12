"""
Same as build_spatial_patches.py, but appends each pixel's own
relative position within the lake's grid (row/(n_rows-1),
col/(n_cols-1), both in [0,1]) to that pixel's static_vals -- giving
CNN-LSTM an explicit per-pixel identity signal, the same hypothesis
already tested and confirmed for the plain LSTM.

Uses the ORIGINAL, unmodified per-lake static_vals as a base -- only
the two new relative-position values are pixel-specific; everything
else about the pipeline (patches, scalar fill logic, subsampling,
target normalization) is identical to the original script.

CRITICAL: uses relative ROW/COL within the reconstructed grid (not
raw lat/lon degrees) -- this matches exactly what row/col already
represent in the original script (pixel_pos_map), avoiding a second,
separate coordinate system.

Only writes to a new patch_cache/latlon/ subdirectory -- never
touches the original, verified patch_cache/ files.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
import gc

DATA_DIR      = "/u/home/elwasila/firelakes/data/model_ready"
PATCH_CACHE_DIR = "/u/home/elwasila/firelakes/data/model_ready/patch_cache/latlon"
os.makedirs(PATCH_CACHE_DIR, exist_ok=True)

VARIABLE = sys.argv[1] if len(sys.argv) > 1 else 'chla'
TARGET_HORIZON = sys.argv[2] if len(sys.argv) > 2 else '1week'
VERSION = sys.argv[3] if len(sys.argv) > 3 else 'with_fire'

assert VARIABLE in ['chla', 'tsm']
assert VERSION in ['with_fire', 'no_fire']

TARGET_BASE = f'{VARIABLE}_next_{TARGET_HORIZON}'
TARGET_ANOMALY = f'{TARGET_BASE}_anomaly'

LOOKBACK = 10
SUBSAMPLE_RATE = 0.15
PATCH_SIZE = 3
PATCH_HALF = PATCH_SIZE // 2
RES = 1.0 / 120.0

np.random.seed(42)
print(f"Building spatial patches + rel lat/lon | Variable: {VARIABLE} | Horizon: {TARGET_HORIZON} | Version: {VERSION}", flush=True)

SPATIAL_COLS_BASE = ['chla_anomaly', 'tsm_anomaly']
SPATIAL_FIRE_COLS = ['ba_weekly_km2']

SCALAR_COLS_BASE = [
    'lswt_anomaly',
    'temp_2m_mean_c_anomaly', 'temp_2m_max_c_anomaly',
    'precip_mm_anomaly', 'runoff_mm_anomaly',
    'surface_runoff_mm_anomaly', 'wind_speed_ms_anomaly',
    'radiation_wm2_anomaly',
]
SCALAR_FIRE_COLS = [
    'ba_prop_catchment', 'days_since_fire',
    'ba_forest_km2', 'ba_grassland_km2', 'ba_shrubland_km2', 'ba_cropland_km2',
    'n_fire_events', 'max_fire_size_km2', 'total_perimeter_km',
]

STATIC_COLS = [
    'depth_avg_m', 'vol_total_km3', 'wshd_area_km2',
    'res_time_days', 'elevation_m', 'lc_tre_usu', 'slp_dg_uav',
]

FIRE_COLS_ALL = SPATIAL_FIRE_COLS + SCALAR_FIRE_COLS
MEDIAN_FILL_COLS = ['precip_mm_anomaly', 'runoff_mm_anomaly', 'surface_runoff_mm_anomaly']

SPATIAL_COLS = SPATIAL_COLS_BASE + (SPATIAL_FIRE_COLS if VERSION == 'with_fire' else [])
SCALAR_COLS  = SCALAR_COLS_BASE + (SCALAR_FIRE_COLS if VERSION == 'with_fire' else [])
N_SPATIAL = len(SPATIAL_COLS)
N_SCALAR = len(SCALAR_COLS)

print(f"Spatial channels: {N_SPATIAL} | Scalar channels: {N_SCALAR} | Patch size: {PATCH_SIZE}x{PATCH_SIZE}", flush=True)
print("NEW: static_vals will have 2 extra dims appended (rel_row, rel_col)", flush=True)

def safe_minmax(arr):
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (0.0, 1.0, True)
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return (vmin, vmax, True)
    return (vmin, vmax, False)

print("Loading deseasonalized train/val/test...", flush=True)
train_df = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_train.parquet'))
val_df   = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_val.parquet'))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_test.parquet'))
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
del train_df, val_df, test_df
gc.collect()

with open(os.path.join(DATA_DIR, f'{VARIABLE}_lake_weights_v2.json')) as f:
    lake_weights = json.load(f)

lake_names = sorted(full_df['lake_name'].unique())
lake_to_idx = {name: i for i, name in enumerate(lake_names)}

ALL_NORM_COLS = SPATIAL_COLS + SCALAR_COLS + [TARGET_ANOMALY]
norm_stats = {}
train_mask_global = (full_df['year'] >= 2002) & (full_df['year'] <= 2018)
for lake in lake_names:
    lt = full_df[train_mask_global & (full_df['lake_name'] == lake)]
    if len(lt) == 0:
        continue
    stats = {}
    for col in ALL_NORM_COLS:
        vmin, vmax, degenerate = safe_minmax(lt[col].values.astype(float))
        stats[col] = (vmin, vmax, degenerate)
    norm_stats[lake] = stats

static_values_by_lake = {}
for lake in lake_names:
    lt = full_df[train_mask_global & (full_df['lake_name'] == lake)]
    if len(lt) == 0:
        continue
    static_values_by_lake[lake] = lt[STATIC_COLS].iloc[0].fillna(0)

static_df = pd.DataFrame(static_values_by_lake).T
static_norm_stats = {}
for col in STATIC_COLS:
    vmin, vmax, degenerate = safe_minmax(static_df[col].values.astype(float))
    static_norm_stats[col] = (vmin, vmax, degenerate)

for lake in norm_stats:
    for col in STATIC_COLS:
        norm_stats[lake][col] = static_norm_stats[col]

def normalize(value, lake, col):
    if lake not in norm_stats or col not in norm_stats[lake]:
        return value
    vmin, vmax, degenerate = norm_stats[lake][col]
    if degenerate:
        return 0.5 if np.isscalar(value) else np.full_like(np.asarray(value, dtype=np.float32), 0.5)
    return (value - vmin) / (vmax - vmin)

splits = {'train': (2002, 2018), 'val': (2019, 2020), 'test': (2021, 2022)}
split_counts = {'train': 0, 'val': 0, 'test': 0}
lake_files = {'train': [], 'val': [], 'test': []}
lake_file_counts = {'train': {}, 'val': {}, 'test': {}}

total_missing_timesteps = 0
total_timesteps_checked = 0

for lake in lake_names:
    print(f"\nProcessing lake: {lake}", flush=True)
    lake_df = full_df[full_df['lake_name'] == lake].copy()
    if len(lake_df) == 0 or lake not in norm_stats:
        print(f"  Skipping {lake} (no data or no norm_stats)", flush=True)
        continue

    lat_min, lon_min = lake_df['lat'].min(), lake_df['lon'].min()
    lake_df['row'] = ((lake_df['lat'] - lat_min) / RES).round().astype(int)
    lake_df['col'] = ((lake_df['lon'] - lon_min) / RES).round().astype(int)
    n_rows = lake_df['row'].max() + 1
    n_cols = lake_df['col'].max() + 1

    lake_df = lake_df.sort_values(['year', 'week'])
    unique_weeks = sorted(lake_df[['year', 'week']].drop_duplicates().apply(tuple, axis=1).tolist())
    week_to_idx = {w: i for i, w in enumerate(unique_weeks)}
    lake_df['week_idx'] = lake_df.set_index(['year', 'week']).index.map(week_to_idx)
    n_weeks_lake = len(unique_weeks)

    print(f"  Grid: {n_rows} x {n_cols} | Weeks: {n_weeks_lake} | Pixels: {lake_df['cell_id'].nunique()}", flush=True)

    lake_week_median = lake_df.groupby('week_idx')[SCALAR_COLS].median()

    spatial_grid = np.full((n_weeks_lake, n_rows, n_cols, N_SPATIAL), np.nan, dtype=np.float32)
    for j, col_name in enumerate(SPATIAL_COLS):
        normed_vals = normalize(lake_df[col_name].values.astype(float), lake, col_name).astype(np.float32)
        spatial_grid[lake_df['week_idx'].values, lake_df['row'].values, lake_df['col'].values, j] = normed_vals
    spatial_grid = np.nan_to_num(spatial_grid, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)

    static_vals_raw = lake_df[STATIC_COLS].iloc[0].fillna(0)
    static_vals_base = np.array([
        normalize(static_vals_raw[col], lake, col) for col in STATIC_COLS
    ], dtype=np.float32)
    static_vals_base = np.nan_to_num(static_vals_base, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)

    lake_idx = lake_to_idx[lake]
    weight = lake_weights.get(lake, 1.0)

    pixel_positions = lake_df[['cell_id', 'row', 'col']].drop_duplicates(subset=['cell_id'])
    pixel_pos_map = dict(zip(
        pixel_positions['cell_id'].astype(int),
        zip(pixel_positions['row'].astype(int), pixel_positions['col'].astype(int))
    ))

    lake_median_norm = {}
    for wk in lake_week_median.index:
        vals = np.array([
            normalize(lake_week_median.loc[wk, col], lake, col) for col in SCALAR_COLS
        ], dtype=np.float16)
        lake_median_norm[wk] = vals

    lake_samples = {'train': [], 'val': [], 'test': []}

    for cell_id, group in lake_df.groupby('cell_id'):
        group = group.sort_values('week_idx').reset_index(drop=True)
        if TARGET_ANOMALY not in group.columns:
            continue
        row, col = pixel_pos_map[int(cell_id)]

        # NEW: this pixel's own relative position within the lake grid,
        # appended to a PER-PIXEL copy of static_vals (never shared
        # across pixels, unlike the original static_vals_base above)
        rel_row = row / (n_rows - 1) if n_rows > 1 else 0.5
        rel_col = col / (n_cols - 1) if n_cols > 1 else 0.5
        static_vals = np.concatenate([
            static_vals_base, np.array([rel_row, rel_col], dtype=np.float16)
        ]).astype(np.float16)

        NON_FIRE_SCALAR_COLS = [c for c in SCALAR_COLS if c not in FIRE_COLS_ALL]

        scalar_by_week = {}
        week_idx_vals = group['week_idx'].values
        scalar_raw = np.zeros((len(group), N_SCALAR), dtype=np.float32)
        for j, col_name in enumerate(SCALAR_COLS):
            if col_name in FIRE_COLS_ALL:
                raw_vals = group[col_name].values.astype(float)
                raw_vals = np.nan_to_num(raw_vals, nan=0.0)
                scalar_raw[:, j] = normalize(raw_vals, lake, col_name).astype(np.float32)
            else:
                scalar_raw[:, j] = normalize(group[col_name].values, lake, col_name).astype(np.float32)

        if len(NON_FIRE_SCALAR_COLS) > 0:
            row_has_nan_arr = group[NON_FIRE_SCALAR_COLS].isna().any(axis=1).values
        else:
            row_has_nan_arr = np.zeros(len(group), dtype=bool)

        for i, wk in enumerate(week_idx_vals):
            scalar_by_week[wk] = (scalar_raw[i].astype(np.float16), not row_has_nan_arr[i])

        target_vals_arr = group[TARGET_ANOMALY].values
        week_idx_arr = group['week_idx'].values
        year_arr = group['year'].values

        for t in range(len(group)):
            target_val = target_vals_arr[t]
            if pd.isna(target_val):
                continue
            week_idx_t = week_idx_arr[t]
            if week_idx_t < LOOKBACK:
                continue

            spatial_patch_seq = np.zeros((LOOKBACK, PATCH_SIZE, PATCH_SIZE, N_SPATIAL), dtype=np.float16)
            scalar_seq = np.zeros((LOOKBACK, N_SCALAR), dtype=np.float16)
            last_valid_scalar = None

            for lb in range(LOOKBACK):
                wk = week_idx_t - LOOKBACK + lb
                r_start, r_end = row - PATCH_HALF, row + PATCH_HALF + 1
                c_start, c_end = col - PATCH_HALF, col + PATCH_HALF + 1
                r_start_clip, r_end_clip = max(r_start, 0), min(r_end, n_rows)
                c_start_clip, c_end_clip = max(c_start, 0), min(c_end, n_cols)
                out_r_start = r_start_clip - r_start
                out_r_end = out_r_start + (r_end_clip - r_start_clip)
                out_c_start = c_start_clip - c_start
                out_c_end = out_c_start + (c_end_clip - c_start_clip)
                spatial_patch_seq[lb, out_r_start:out_r_end, out_c_start:out_c_end, :] = \
                    spatial_grid[wk, r_start_clip:r_end_clip, c_start_clip:c_end_clip, :]

                total_timesteps_checked += 1
                if wk in scalar_by_week:
                    vals, is_complete = scalar_by_week[wk]
                    if is_complete:
                        vals = np.nan_to_num(vals, nan=0.0)
                        scalar_seq[lb] = vals
                        last_valid_scalar = vals.copy()
                        continue

                total_missing_timesteps += 1
                filled = np.zeros(N_SCALAR, dtype=np.float16)
                for j, col_name in enumerate(SCALAR_COLS):
                    if col_name in FIRE_COLS_ALL:
                        filled[j] = 0.0
                    elif col_name in MEDIAN_FILL_COLS:
                        val = lake_median_norm.get(wk, np.zeros(N_SCALAR, dtype=np.float16))[j]
                        filled[j] = val if not np.isnan(val) else 0.0
                    else:
                        if last_valid_scalar is not None and not np.isnan(last_valid_scalar[j]):
                            filled[j] = last_valid_scalar[j]
                        else:
                            val = lake_median_norm.get(wk, np.zeros(N_SCALAR, dtype=np.float16))[j]
                            filled[j] = val if not np.isnan(val) else 0.0
                scalar_seq[lb] = filled
                if np.isnan(scalar_seq[lb]).any():
                    scalar_seq[lb] = np.nan_to_num(scalar_seq[lb], nan=0.0)

            if np.random.random() > SUBSAMPLE_RATE:
                continue

            row_year = year_arr[t]
            target_norm = np.float16(normalize(target_val, lake, TARGET_ANOMALY))

            for split_name, (sy, ey) in splits.items():
                if sy <= row_year <= ey:
                    lake_samples[split_name].append((
                        spatial_patch_seq, scalar_seq, static_vals, lake_idx, target_norm, weight
                    ))
                    break

    for split_name in ['train', 'val', 'test']:
        if len(lake_samples[split_name]) == 0:
            continue
        fname = f'{VERSION}_{VARIABLE}_{TARGET_HORIZON}_{lake}_{split_name}.pkl'
        fpath = os.path.join(PATCH_CACHE_DIR, fname)
        with open(fpath, 'wb') as f:
            pickle.dump(lake_samples[split_name], f)
        lake_files[split_name].append(fname)
        lake_file_counts[split_name][fname] = len(lake_samples[split_name])
        split_counts[split_name] += len(lake_samples[split_name])

    del spatial_grid, lake_df, lake_samples
    gc.collect()
    pct_missing_so_far = (total_missing_timesteps / max(total_timesteps_checked, 1)) * 100
    print(f"  Saved. Running totals — Train: {split_counts['train']:,} | "
          f"Val: {split_counts['val']:,} | Test: {split_counts['test']:,} | "
          f"Missing-timestep rate so far: {pct_missing_so_far:.2f}%", flush=True)

index = {
    'lake_files': lake_files,
    'lake_to_idx': lake_to_idx,
    'norm_stats': norm_stats,
    'n_lakes': len(lake_names),
    'split_counts': split_counts,
    'n_spatial': N_SPATIAL, 'n_scalar': N_SCALAR,
    'lake_file_counts': lake_file_counts,
}
index_path = os.path.join(PATCH_CACHE_DIR, f'{VERSION}_{VARIABLE}_{TARGET_HORIZON}_INDEX.pkl')
with open(index_path, 'wb') as f:
    pickle.dump(index, f)

final_pct_missing = (total_missing_timesteps / max(total_timesteps_checked, 1)) * 100
print(f"\nSaved index to: {index_path}", flush=True)
print(f"Final — Train: {split_counts['train']:,} | Val: {split_counts['val']:,} | Test: {split_counts['test']:,}", flush=True)
print(f"Final missing-timestep rate (all columns combined): {final_pct_missing:.2f}%", flush=True)
