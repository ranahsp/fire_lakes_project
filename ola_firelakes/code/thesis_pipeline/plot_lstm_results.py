import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "/u/home/elwasila/firelakes/models/results/LSTM"
PLOT_DIR = "/u/home/elwasila/firelakes/models/results/LSTM/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

HORIZONS = ['1week', '1month', '3month']
HORIZON_LABELS = {'1week': '1 wk', '1month': '1 mo', '3month': '3 mo'}
VARIABLES = ['chla', 'tsm']

# ── Manually corrected chla 3-month values (6 seeds combined) ──
CHLA_3MONTH_CORRECTED = {
    'With fire':    {'r2_mean': 0.1458, 'r2_std': 0.0257,
                      'rmse_mean': np.mean([7.1726,6.8828,7.0594,7.1738,7.2064,7.0789]),
                      'rmse_std': np.std([7.1726,6.8828,7.0594,7.1738,7.2064,7.0789])},
    'Without fire': {'r2_mean': 0.0838, 'r2_std': 0.0779,
                      'rmse_mean': np.mean([7.3826,7.6221,7.5488,7.4150,6.6671,7.4183]),
                      'rmse_std': np.std([7.3826,7.6221,7.5488,7.4150,6.6671,7.4183])},
}

all_rows = []
for var in VARIABLES:
    for hz in HORIZONS:
        fpath = os.path.join(RESULTS_DIR, f'lstm_multiseed_comparison_{var}_{hz}.csv')
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df['variable'] = var
            df['Horizon'] = hz  # force-fill, fixes the NaN issue for 1week

            # Override chla 3month with corrected 6-seed values
            if var == 'chla' and hz == '3month':
                for i, row in df.iterrows():
                    corrected = CHLA_3MONTH_CORRECTED[row['Version']]
                    df.at[i, 'R2_mean'] = corrected['r2_mean']
                    df.at[i, 'R2_std'] = corrected['r2_std']
                    df.at[i, 'RMSE_mean'] = corrected['rmse_mean']
                    df.at[i, 'RMSE_std'] = corrected['rmse_std']

            all_rows.append(df)

master = pd.concat(all_rows, ignore_index=True)
master.to_csv(os.path.join(RESULTS_DIR, 'lstm_master_summary.csv'), index=False)
print("Master summary (corrected):")
print(master.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, var in zip(axes, VARIABLES):
    sub = master[master['variable'] == var]
    for version, color, marker in [('With fire', 'firebrick', 'o'), ('Without fire', 'steelblue', 's')]:
        vsub = sub[sub['Version'] == version].set_index('Horizon').reindex(HORIZONS).reset_index()
        ax.errorbar(
            [HORIZON_LABELS[h] for h in vsub['Horizon']],
            vsub['R2_mean'], yerr=vsub['R2_std'],
            label=version, color=color, marker=marker, capsize=4, linewidth=2, markersize=8
        )
    ax.set_title(f'{var.upper()} — LSTM prediction skill vs. horizon')
    ax.set_xlabel('Prediction horizon')
    ax.set_ylabel('Test R²')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, 'lstm_r2_vs_horizon.png')
plt.savefig(plot_path, dpi=150)
print(f"\nSaved plot: {plot_path}")
