# %%
import pandas as pd
import numpy as np
import os
import json
import gc

DATA_DIR = "/u/home/elwasila/firelakes/data"
FEATURES_PATH = os.path.join(DATA_DIR, "features", "all_features_merged.parquet")
OUT_DIR = "/u/home/elwasila/firelakes/data/model_ready"
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE_LAKES = ['buffalo']  # 0% usable chla_next and tsm_next — confirmed

GAP_YEARS = (2012, 2015)  # uniform removal per Stroppiana's instruction

TRAIN_YEARS = (2002, 2018)
VAL_YEARS   = (2019, 2020)
TEST_YEARS  = (2021, 2022)

print("✅ Configuration loaded")
print(f"   Features path: {FEATURES_PATH}")
print(f"   Output dir: {OUT_DIR}")

# %%
raw_cache_path = os.path.join(OUT_DIR, 'raw_loaded.parquet')

if os.path.exists(raw_cache_path):
    print("✅ Found cached raw load — loading from disk")
    df = pd.read_parquet(raw_cache_path)
else:
    print("⏳ Loading parquet file...")
    df = pd.read_parquet(FEATURES_PATH)
    df.to_parquet(raw_cache_path)
    print(f"💾 Cached to: {raw_cache_path}")

print(f"✅ Data ready: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# %%
excluded_cache_path = os.path.join(OUT_DIR, 'excluded_lakes_applied.parquet')

if os.path.exists(excluded_cache_path):
    print("✅ Found cached exclusion step — loading from disk")
    df_clean = pd.read_parquet(excluded_cache_path)
else:
    print(f"⏳ Excluding lakes: {EXCLUDE_LAKES}...")
    df_clean = df[~df['lake_name'].isin(EXCLUDE_LAKES)]
    df_clean.to_parquet(excluded_cache_path)
    del df
    gc.collect()
    print(f"💾 Cached to: {excluded_cache_path}")

print(f"✅ After exclusion: {df_clean.shape}, {df_clean['lake_name'].nunique()} lakes")

# %%
gap_removed_path = os.path.join(OUT_DIR, 'gap_removed.parquet')

if os.path.exists(gap_removed_path):
    print("✅ Found cached gap-removal step — loading from disk")
    df_no_gap = pd.read_parquet(gap_removed_path)
else:
    print(f"⏳ Dropping years {GAP_YEARS[0]}-{GAP_YEARS[1]} for all lakes...")

    before_rows = len(df_clean)
    keep_mask = ~df_clean['year'].between(GAP_YEARS[0], GAP_YEARS[1])
    df_no_gap = df_clean[keep_mask]
    after_rows = len(df_no_gap)

    df_no_gap.to_parquet(gap_removed_path)
    del df_clean, keep_mask
    gc.collect()
    print(f"💾 Cached to: {gap_removed_path}")
    print(f"   Rows before: {before_rows:,} | after: {after_rows:,} "
          f"| removed: {before_rows - after_rows:,}")

print(f"✅ Gap-removed dataset ready: {df_no_gap.shape}")

# %%
horizons_cache_path = os.path.join(OUT_DIR, 'multi_horizon_targets.parquet')

if os.path.exists(horizons_cache_path):
    print("✅ Found cached multi-horizon targets — loading from disk")
    df_horizons = pd.read_parquet(horizons_cache_path)
else:
    print("⏳ Engineering multi-horizon targets per pixel...")

    df_sorted = df_no_gap.sort_values(['lake_name', 'cell_id', 'year', 'week'])

    HORIZON_WEEKS = {
        '1week': 1, '1month': 4, '3month': 13,
        '6month': 26, '1year': 52, '2year': 104,
    }

    for target_base in ['chla', 'tsm']:
        for horizon_name, n_weeks in HORIZON_WEEKS.items():
            col_name = f'{target_base}_next_{horizon_name}'
            df_sorted[col_name] = (
                df_sorted.groupby(['lake_name', 'cell_id'])[target_base]
                .shift(-n_weeks)
            )
            print(f"   ✅ {col_name} created")

    df_horizons = df_sorted
    df_horizons.to_parquet(horizons_cache_path)
    del df_no_gap, df_sorted
    gc.collect()
    print(f"💾 Cached to: {horizons_cache_path}")

print(f"\n✅ Multi-horizon targets ready: {df_horizons.shape}")

# %%
windowed_cache_path = os.path.join(OUT_DIR, 'windowed_horizon_targets.parquet')

if os.path.exists(windowed_cache_path):
    print("✅ Found cached windowed targets — loading from disk")
    df_windowed = pd.read_parquet(windowed_cache_path)
else:
    print("⏳ Computing windowed-average targets for 1yr/2yr horizons...")

    df_windowed = df_horizons
    LONG_HORIZONS = {'1year': 52, '2year': 104}
    WINDOW_HALF_WIDTH = 2

    for target_base in ['chla', 'tsm']:
        for horizon_name, n_weeks in LONG_HORIZONS.items():
            col_name = f'{target_base}_next_{horizon_name}_windowed'
            grouped = df_windowed.groupby(['lake_name', 'cell_id'])[target_base]
            shifted_cols = [
                grouped.shift(-(n_weeks + offset))
                for offset in range(-WINDOW_HALF_WIDTH, WINDOW_HALF_WIDTH + 1)
            ]
            df_windowed[col_name] = pd.concat(shifted_cols, axis=1).mean(axis=1)
            print(f"   ✅ {col_name} created")

    df_windowed.to_parquet(windowed_cache_path)
    del df_horizons
    gc.collect()
    print(f"💾 Cached to: {windowed_cache_path}")

print(f"\n✅ Windowed targets ready: {df_windowed.shape}")

# %%
safeguarded_path = os.path.join(OUT_DIR, 'lag_safeguarded.parquet')

if os.path.exists(safeguarded_path):
    print("✅ Found cached safeguard step — loading from disk")
    df_safe = pd.read_parquet(safeguarded_path)
else:
    print("⏳ Checking for lag discontinuities caused by gap removal...")

    df_safe = df_windowed.sort_values(['lake_name', 'cell_id', 'year', 'week'])
    df_safe['abs_week'] = df_safe['year'] * 52 + df_safe['week']

    prev_abs_week = df_safe.groupby(['lake_name', 'cell_id'])['abs_week'].shift(1)
    week_gap = df_safe['abs_week'] - prev_abs_week
    broken_lag = (week_gap > 1) | week_gap.isna()

    n_broken = broken_lag.sum()
    print(f"⚠️  Found {n_broken:,} rows where lag features are invalid "
          f"due to the gap (setting to NaN)")

    for col in ['chla_lag1', 'tsm_lag1', 'lswt_lag1']:
        if col in df_safe.columns:
            df_safe.loc[broken_lag, col] = np.nan

    df_safe = df_safe.drop(columns=['abs_week'])
    df_safe.to_parquet(safeguarded_path)
    del df_windowed
    gc.collect()
    print(f"💾 Cached to: {safeguarded_path}")

print(f"\n✅ Lag-safeguarded dataset ready: {df_safe.shape}")

# %%
train_path = os.path.join(OUT_DIR, 'train.parquet')
val_path   = os.path.join(OUT_DIR, 'val.parquet')
test_path  = os.path.join(OUT_DIR, 'test.parquet')

if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
    print("✅ Found cached splits — loading from disk")
    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)
    test_df  = pd.read_parquet(test_path)
else:
    print("⏳ Building time-based splits...")

    def split_by_year(data, start, end):
        return data[(data['year'] >= start) & (data['year'] <= end)]

    train_df = split_by_year(df_safe, *TRAIN_YEARS)
    val_df   = split_by_year(df_safe, *VAL_YEARS)
    test_df  = split_by_year(df_safe, *TEST_YEARS)

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    test_df.to_parquet(test_path)
    print(f"💾 Saved splits to: {OUT_DIR}")

print(f"\n✅ Train {TRAIN_YEARS}: {len(train_df):,} rows")
print(f"✅ Val   {VAL_YEARS}: {len(val_df):,} rows")
print(f"✅ Test  {TEST_YEARS}: {len(test_df):,} rows")

# %%
chla_weights_path = os.path.join(OUT_DIR, 'chla_lake_weights_v2.json')
tsm_weights_path  = os.path.join(OUT_DIR, 'tsm_lake_weights_v2.json')

if os.path.exists(chla_weights_path) and os.path.exists(tsm_weights_path):
    print("✅ Found cached weights (v2) — loading from disk")
    with open(chla_weights_path) as f:
        chla_weights = json.load(f)
    with open(tsm_weights_path) as f:
        tsm_weights = json.load(f)
else:
    print("⏳ Computing weights on gap-removed, lag-safeguarded train data...")

    def compute_lake_weights(train_data, target_col):
        usable = train_data[train_data[target_col].notna()]
        counts = usable.groupby('lake_name', observed=True).size()
        counts = counts[counts > 0]
        raw_weights = 1 / np.sqrt(counts)
        norm_weights = raw_weights / raw_weights.mean()
        return norm_weights.to_dict()

    chla_weights = compute_lake_weights(train_df, 'chla_next_1week')
    tsm_weights  = compute_lake_weights(train_df, 'tsm_next_1week')

    with open(chla_weights_path, 'w') as f:
        json.dump(chla_weights, f)
    with open(tsm_weights_path, 'w') as f:
        json.dump(tsm_weights, f)
    print(f"💾 Saved weights (v2) to: {OUT_DIR}")

print("\nCHLA lake weights (v2, sorted lowest to highest):")
print(pd.Series(chla_weights).sort_values().to_string())

# %%
print("\n" + "="*60)
print("✅ DATA PREP COMPLETE")
print("="*60)
for fname in ['train.parquet', 'val.parquet', 'test.parquet',
              'chla_lake_weights_v2.json', 'tsm_lake_weights_v2.json']:
    fpath = os.path.join(OUT_DIR, fname)
    status = "✅" if os.path.exists(fpath) else "❌ MISSING"
    print(f"   {status} {fname}")

# %%
import os

OUT_DIR = "/u/home/elwasila/firelakes/data/model_ready"
print("Files in model_ready:")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    size_mb = os.path.getsize(fpath) / (1024*1024)
    print(f"   {f:<40} {size_mb:>10.1f} MB")

# %%
print(f"Train: {train_df.shape}")
print(f"Val:   {val_df.shape}")
print(f"Test:  {test_df.shape}")

target_cols = [c for c in train_df.columns if '_next_' in c]
print(f"\nTarget columns ({len(target_cols)}):")
for c in sorted(target_cols):
    print(f"   {c}")

# %%
climatology_path = "/u/home/elwasila/firelakes/data/model_ready/seasonal_climatology.csv"

DESEASON_VARS = [
    'chla', 'tsm', 'lswt',
    'temp_2m_mean_c', 'temp_2m_max_c',
    'precip_mm', 'runoff_mm', 'surface_runoff_mm',
    'wind_speed_ms', 'radiation_wm2',
]

if os.path.exists(climatology_path):
    print("✅ Found cached climatology — loading from disk")
    climatology = pd.read_csv(climatology_path)
else:
    print("⏳ Computing seasonal climatology from TRAINING data only...")

    climatology = (
        train_df.groupby(['lake_name', 'cell_id', 'week'])[DESEASON_VARS]
        .median()
        .reset_index()
    )
    climatology.columns = ['lake_name', 'cell_id', 'week'] + \
        [f'{v}_climatology' for v in DESEASON_VARS]

    climatology.to_csv(climatology_path, index=False)
    print(f"💾 Saved to: {climatology_path}")

print(f"✅ Climatology ready: {climatology.shape}")
print(climatology.head())

# %%
deseasonalized_path_prefix = "/u/home/elwasila/firelakes/data/model_ready/deseason"

def deseasonalize_split(df, climatology, split_name):
    cache_path = f"{deseasonalized_path_prefix}_{split_name}.parquet"
    if os.path.exists(cache_path):
        print(f"✅ {split_name}: found cached deseasonalized split")
        return pd.read_parquet(cache_path)

    print(f"⏳ {split_name}: applying anomaly transform...")
    merged = df.merge(climatology, on=['lake_name', 'cell_id', 'week'], how='left')

    for v in DESEASON_VARS:
        clim_col = f'{v}_climatology'
        merged[f'{v}_anomaly'] = merged[v] - merged[clim_col]

    # Lag features are just prior-week raw values — deseasonalize them
    # using the SAME climatology (their own week's climatology, since
    # chla_lag1 already represents week t-1's value, and week_of_year
    # for t-1 differs by 1 — but our merge above is keyed to row t's week.
    # We approximate using a shifted climatology lookup:
    for lag_col, base_var in [('chla_lag1', 'chla'), ('tsm_lag1', 'tsm'), ('lswt_lag1', 'lswt')]:
        prev_week = ((merged['week'] - 2) % 52) + 1  # week t-1's week-of-year
        clim_lookup = climatology.set_index(['lake_name', 'cell_id', 'week'])[f'{base_var}_climatology']
        keys = list(zip(merged['lake_name'], merged['cell_id'], prev_week))
        merged[f'{lag_col}_anomaly'] = merged[lag_col] - clim_lookup.reindex(keys).values

    merged.to_parquet(cache_path)
    print(f"💾 {split_name}: cached to {cache_path}")
    return merged

train_deseason = deseasonalize_split(train_df, climatology, 'train')
val_deseason   = deseasonalize_split(val_df, climatology, 'val')
test_deseason  = deseasonalize_split(test_df, climatology, 'test')

print(f"\n✅ Deseasonalized splits ready")
print(f"Train: {train_deseason.shape}")

# %%
targets_deseason_path_prefix = "/u/home/elwasila/firelakes/data/model_ready/deseason_targets"

TARGET_HORIZONS = {
    'chla_next_1week': ('chla', 1), 'tsm_next_1week': ('tsm', 1),
    'chla_next_1month': ('chla', 4), 'tsm_next_1month': ('tsm', 4),
}
# (extend this dict with more horizons as needed — shown here for the
#  primary 1-week target plus 1-month as an example)

def deseasonalize_targets(df, climatology, split_name):
    cache_path = f"{targets_deseason_path_prefix}_{split_name}.parquet"
    if os.path.exists(cache_path):
        print(f"✅ {split_name}: found cached deseasonalized targets")
        return pd.read_parquet(cache_path)

    print(f"⏳ {split_name}: deseasonalizing targets...")
    result = df.copy()

    for target_col, (base_var, n_weeks) in TARGET_HORIZONS.items():
        if target_col not in result.columns:
            continue
        future_week = ((result['week'] - 1 + n_weeks) % 52) + 1
        clim_lookup = climatology.set_index(['lake_name', 'cell_id', 'week'])[f'{base_var}_climatology']
        keys = list(zip(result['lake_name'], result['cell_id'], future_week))
        result[f'{target_col}_anomaly'] = result[target_col] - clim_lookup.reindex(keys).values
        # Keep the future_week too, needed later to re-seasonalize predictions
        result[f'{target_col}_future_week'] = future_week

    result.to_parquet(cache_path)
    print(f"💾 {split_name}: cached to {cache_path}")
    return result

train_final = deseasonalize_targets(train_deseason, climatology, 'train')
val_final   = deseasonalize_targets(val_deseason, climatology, 'val')
test_final  = deseasonalize_targets(test_deseason, climatology, 'test')

print(f"\n✅ Final deseasonalized datasets ready")
print(f"Train: {train_final.shape}")

# %%
# %%
targets_deseason_path_prefix = "/u/home/elwasila/firelakes/data/model_ready/deseason_targets_v2"

TARGET_HORIZONS = {
    'chla_next_1week':  ('chla', 1),
    'chla_next_1month': ('chla', 4),
    'chla_next_3month': ('chla', 13),
    'chla_next_6month': ('chla', 26),
    'chla_next_1year':  ('chla', 52),
    'chla_next_2year':  ('chla', 104),
    'tsm_next_1week':   ('tsm', 1),
    'tsm_next_1month':  ('tsm', 4),
    'tsm_next_3month':  ('tsm', 13),
    'tsm_next_6month':  ('tsm', 26),
    'tsm_next_1year':   ('tsm', 52),
    'tsm_next_2year':   ('tsm', 104),
}

def deseasonalize_targets_v2(df, climatology, split_name):
    cache_path = f"{targets_deseason_path_prefix}_{split_name}.parquet"
    if os.path.exists(cache_path):
        print(f"✅ {split_name}: found cached deseasonalized targets (v2)")
        return pd.read_parquet(cache_path)

    print(f"⏳ {split_name}: deseasonalizing all horizon targets...")
    result = df.copy()

    for target_col, (base_var, n_weeks) in TARGET_HORIZONS.items():
        if target_col not in result.columns:
            print(f"   ⚠️  {target_col} not found in dataframe — skipping")
            continue
        future_week = ((result['week'] - 1 + n_weeks) % 52) + 1
        clim_lookup = climatology.set_index(['lake_name', 'cell_id', 'week'])[f'{base_var}_climatology']
        keys = list(zip(result['lake_name'], result['cell_id'], future_week))
        result[f'{target_col}_anomaly'] = result[target_col] - clim_lookup.reindex(keys).values
        result[f'{target_col}_future_week'] = future_week
        print(f"   ✅ {target_col}_anomaly created")

    result.to_parquet(cache_path)
    print(f"💾 {split_name}: cached to {cache_path}")
    return result

train_final_v2 = deseasonalize_targets_v2(train_deseason, climatology, 'train')
val_final_v2   = deseasonalize_targets_v2(val_deseason, climatology, 'val')
test_final_v2  = deseasonalize_targets_v2(test_deseason, climatology, 'test')

print(f"\n✅ All-horizon deseasonalized datasets ready")
print(f"Train: {train_final_v2.shape}")


