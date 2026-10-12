import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR    = "/u/home/elwasila/firelakes/data/model_ready"
RESULTS_DIR = "/u/home/elwasila/firelakes/models/results/RF"
os.makedirs(RESULTS_DIR, exist_ok=True)

HORIZONS = ['1week', '1month', '3month', '6month', '1year', '2year']
VARIABLE = 'chla'

FEATURE_COLS_BASE = [
    'chla_lag1_anomaly', 'tsm_lag1_anomaly', 'lswt_lag1_anomaly',
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

FEATURE_COLS_WITH_FIRE = FEATURE_COLS_BASE + FIRE_COLS
FEATURE_COLS_NO_FIRE   = FEATURE_COLS_BASE

print(f"Variable: {VARIABLE}")
print(f"Horizons: {HORIZONS}")

print("\nLoading deseasonalized train/val/test (v2)...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_train.parquet'))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, 'deseason_targets_v2_test.parquet'))

weight_file = 'chla_lake_weights_v2.json' if VARIABLE == 'chla' else 'tsm_lake_weights_v2.json'
with open(os.path.join(DATA_DIR, weight_file)) as f:
    lake_weights = json.load(f)

climatology = pd.read_csv(os.path.join(DATA_DIR, 'seasonal_climatology.csv'))
clim_col = f'{VARIABLE}_climatology'
clim_lookup = climatology.set_index(['lake_name', 'cell_id', 'week'])[clim_col]

def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)

def prep(data, target_anomaly_col, feature_cols):
    data = data[data[target_anomaly_col].notna()]
    X = data[feature_cols]
    y = data[target_anomaly_col]
    w = data['lake_name'].map(lake_weights).fillna(1.0).values
    return X, y, w, data

def run_experiment(target_base, feature_cols, version_name):
    target_anomaly = f'{target_base}_anomaly'
    target_future_week = f'{target_base}_future_week'

    X_train, y_train, w_train, _        = prep(train_df, target_anomaly, feature_cols)
    X_test,  y_test,  w_test,  test_meta = prep(test_df, target_anomaly, feature_cols)

    rf = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5,
        n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train, sample_weight=w_train)
    test_pred_anomaly = rf.predict(X_test)

    keys = list(zip(test_meta['lake_name'], test_meta['cell_id'], test_meta[target_future_week]))
    clim_values = clim_lookup.reindex(keys).values

    test_pred_raw = test_pred_anomaly + clim_values
    test_true_raw = y_test.values + clim_values

    rmse = np.sqrt(mean_squared_error(test_true_raw, test_pred_raw))
    mae  = mean_absolute_error(test_true_raw, test_pred_raw)
    r2   = r2_score(test_true_raw, test_pred_raw)
    nse_val = nse(test_true_raw, test_pred_raw)

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

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'nse': nse_val, 'per_lake': per_lake_df}

all_results = []
per_lake_deltas = {}

for horizon in HORIZONS:
    target_base = f'{VARIABLE}_next_{horizon}'
    print(f"\n{'='*60}")
    print(f"HORIZON: {horizon}")
    print(f"{'='*60}")

    if f'{target_base}_anomaly' not in train_df.columns:
        print(f"Warning: {target_base}_anomaly not found - skipping this horizon")
        continue

    with_fire = run_experiment(target_base, FEATURE_COLS_WITH_FIRE, 'With fire')
    no_fire   = run_experiment(target_base, FEATURE_COLS_NO_FIRE, 'Without fire')

    print(f"With fire    - RMSE: {with_fire['rmse']:.4f} | R2: {with_fire['r2']:.4f}")
    print(f"Without fire - RMSE: {no_fire['rmse']:.4f} | R2: {no_fire['r2']:.4f}")

    all_results.append({'horizon': horizon, 'version': 'With fire', **{k: with_fire[k] for k in ['rmse','mae','r2','nse']}})
    all_results.append({'horizon': horizon, 'version': 'Without fire', **{k: no_fire[k] for k in ['rmse','mae','r2','nse']}})

    delta = no_fire['per_lake'].merge(with_fire['per_lake'], on='lake_name', suffixes=('_no_fire', '_with_fire'))
    delta['delta_rmse'] = delta['rmse_no_fire'] - delta['rmse_with_fire']
    delta['horizon'] = horizon
    per_lake_deltas[horizon] = delta

    mean_delta = delta['delta_rmse'].mean()
    n_positive = (delta['delta_rmse'] > 0).sum()
    n_total = len(delta)
    print(f"Unweighted mean delta_rmse across lakes: {mean_delta:.4f} "
          f"({n_positive}/{n_total} lakes show fire helping)")

comparison_df = pd.DataFrame(all_results)
comparison_df.to_csv(os.path.join(RESULTS_DIR, f'rf_multihorizon_comparison_{VARIABLE}.csv'), index=False)

all_deltas_df = pd.concat(per_lake_deltas.values(), ignore_index=True)
all_deltas_df.to_csv(os.path.join(RESULTS_DIR, f'rf_multihorizon_per_lake_delta_{VARIABLE}.csv'), index=False)

print(f"\n{'='*60}")
print("SUMMARY ACROSS ALL HORIZONS")
print(f"{'='*60}")
print(comparison_df.to_string(index=False))

print(f"\nMulti-horizon ablation complete. Results saved to {RESULTS_DIR}")
