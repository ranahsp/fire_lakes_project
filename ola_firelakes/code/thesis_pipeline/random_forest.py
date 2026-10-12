# ============================================================
# Random Forest — Fire vs No-Fire Ablation Study
# ============================================================

import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR    = "/u/home/elwasila/firelakes/data/model_ready"
RESULTS_DIR = "/u/home/elwasila/firelakes/models/results/RF"
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_BASE = 'chla_next_1week'
TARGET_ANOMALY = f'{TARGET_BASE}_anomaly'
TARGET_FUTURE_WEEK = f'{TARGET_BASE}_future_week'

FEATURE_COLS_WITH_FIRE = [
    'chla_lag1_anomaly', 'tsm_lag1_anomaly', 'lswt_lag1_anomaly',
    'ba_weekly_km2', 'ba_prop_catchment',
    'ba_forest_km2', 'ba_grassland_km2',
    'ba_shrubland_km2', 'ba_cropland_km2',
    'days_since_fire', 'watershed_area_km2',
    'n_fire_events', 'max_fire_size_km2', 'total_perimeter_km',
    'temp_2m_mean_c_anomaly', 'temp_2m_max_c_anomaly',
    'precip_mm_anomaly', 'runoff_mm_anomaly',
    'surface_runoff_mm_anomaly', 'wind_speed_ms_anomaly',
    'radiation_wm2_anomaly',
    'depth_avg_m', 'vol_total_km3', 'wshd_area_km2',
    'res_time_days', 'elevation_m', 'lc_tre_usu', 'slp_dg_uav',
]

FIRE_COLS = [
    'ba_weekly_km2', 'ba_prop_catchment',
    'ba_forest_km2', 'ba_grassland_km2',
    'ba_shrubland_km2', 'ba_cropland_km2',
    'days_since_fire', 'watershed_area_km2',
    'n_fire_events', 'max_fire_size_km2', 'total_perimeter_km',
]

FEATURE_COLS_NO_FIRE = [c for c in FEATURE_COLS_WITH_FIRE if c not in FIRE_COLS]

print(f"With-fire feature count: {len(FEATURE_COLS_WITH_FIRE)}")
print(f"No-fire feature count:   {len(FEATURE_COLS_NO_FIRE)}")

# ── Load data ────────────────────────────────────────────────
print("\nLoading deseasonalized train/val/test...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_train.parquet'))
val_df   = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_val.parquet'))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_test.parquet'))

with open(os.path.join(DATA_DIR, 'chla_lake_weights_v2.json')) as f:
    lake_weights = json.load(f)

with open(os.path.join(DATA_DIR, 'seasonal_climatology.csv')) as f:
    pass
climatology = pd.read_csv(os.path.join(DATA_DIR, 'seasonal_climatology.csv'))

def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)

def prep(data, feature_cols):
    data = data[data[TARGET_ANOMALY].notna()]
    X = data[feature_cols]
    y = data[TARGET_ANOMALY]
    w = data['lake_name'].map(lake_weights).fillna(1.0).values
    return X, y, w, data

def run_experiment(version_name, feature_cols):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {version_name}")
    print(f"{'='*60}")

    X_train, y_train, w_train, _        = prep(train_df, feature_cols)
    X_val,   y_val,   w_val,   _        = prep(val_df, feature_cols)
    X_test,  y_test,  w_test,  test_meta = prep(test_df, feature_cols)

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5,
        n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train, sample_weight=w_train)

    val_pred_anomaly = rf.predict(X_val)
    test_pred_anomaly = rf.predict(X_test)

    # ── Re-seasonalize predictions back to raw units ──────────
    clim_col = 'chla_climatology'
    clim_lookup = climatology.set_index(['lake_name', 'cell_id', 'week'])[clim_col]
    keys = list(zip(test_meta['lake_name'], test_meta['cell_id'], test_meta[TARGET_FUTURE_WEEK]))
    clim_values = clim_lookup.reindex(keys).values

    test_pred_raw  = test_pred_anomaly + clim_values
    test_true_raw  = y_test.values + clim_values  # reconstruct raw true value

    rmse = np.sqrt(mean_squared_error(test_true_raw, test_pred_raw))
    mae  = mean_absolute_error(test_true_raw, test_pred_raw)
    r2   = r2_score(test_true_raw, test_pred_raw)
    nse_val = nse(test_true_raw, test_pred_raw)

    print(f"RMSE (raw units): {rmse:.4f}")
    print(f"MAE  (raw units): {mae:.4f}")
    print(f"R²:               {r2:.4f}")
    print(f"NSE:              {nse_val:.4f}")

    # ── Per-lake metrics ───────────────────────────────────────
    results_df = pd.DataFrame({
        'lake_name': test_meta['lake_name'].values,
        'y_true': test_true_raw, 'y_pred': test_pred_raw,
    })
    per_lake = []
    for lake, group in results_df.groupby('lake_name'):
        if len(group) < 5:
            continue
        lake_rmse = np.sqrt(mean_squared_error(group['y_true'], group['y_pred']))
        per_lake.append({'lake_name': lake, 'n': len(group), 'rmse': lake_rmse})
    per_lake_df = pd.DataFrame(per_lake)

    return {
        'version': version_name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'nse': nse_val,
        'per_lake': per_lake_df, 'model': rf,
    }

# ── Run both experiments ──────────────────────────────────────
result_with_fire = run_experiment('With Fire', FEATURE_COLS_WITH_FIRE)
result_no_fire    = run_experiment('Without Fire', FEATURE_COLS_NO_FIRE)

# ── Comparison table ───────────────────────────────────────────
comparison = pd.DataFrame([
    {'Model': 'RF', 'Version': 'With fire', 'RMSE': result_with_fire['rmse'],
     'MAE': result_with_fire['mae'], 'R2': result_with_fire['r2'], 'NSE': result_with_fire['nse']},
    {'Model': 'RF', 'Version': 'Without fire', 'RMSE': result_no_fire['rmse'],
     'MAE': result_no_fire['mae'], 'R2': result_no_fire['r2'], 'NSE': result_no_fire['nse']},
])
print("\n" + "="*60)
print("COMPARISON TABLE")
print("="*60)
print(comparison.to_string(index=False))

# ── Per-lake delta ─────────────────────────────────────────────
delta = result_no_fire['per_lake'].merge(
    result_with_fire['per_lake'], on='lake_name', suffixes=('_no_fire', '_with_fire'))
delta['delta_rmse'] = delta['rmse_no_fire'] - delta['rmse_with_fire']
delta = delta.sort_values('delta_rmse', ascending=False)

print("\n" + "="*60)
print("PER-LAKE DELTA (positive = fire data helps)")
print("="*60)
print(delta.to_string(index=False))

# ── Save everything ─────────────────────────────────────────────
comparison.to_csv(os.path.join(RESULTS_DIR, 'rf_ablation_comparison.csv'), index=False)
delta.to_csv(os.path.join(RESULTS_DIR, 'rf_ablation_per_lake_delta.csv'), index=False)
with open(os.path.join(RESULTS_DIR, 'rf_with_fire.pkl'), 'wb') as f:
    pickle.dump(result_with_fire['model'], f)
with open(os.path.join(RESULTS_DIR, 'rf_no_fire.pkl'), 'wb') as f:
    pickle.dump(result_no_fire['model'], f)

print(f"\n✅ Ablation study complete. Results saved to {RESULTS_DIR}")