"""
plot_comparison.py
------------------
Creates a TDMPC vs O2 comparison plot for a given task.

Reads CSV files from wandb_data/ that match:
  {task}_{tdmpc_exp}_seed*_history.csv   → red solid  (TDMPC baseline, full run)
  {task}_{o2_exp}_seed*_history.csv      → red dashed (O2 TDMPC phase) + blue solid (O2 phase)

The O2 runs use the `phase` column:
  phase 1/2 → TDMPC phase (red dashed)
  phase 3   → O2 phase    (blue solid)

Shaded areas are ±1 std across seeds.

Usage:
    python scripts/plot_comparison.py --task cartpole-swingup
    python scripts/plot_comparison.py --task walker-walk --tdmpc-exp TDMPC_normal --o2-exp o2_phased
    python scripts/plot_comparison.py --task cartpole-swingup --smooth 5 --out plots/cartpole_comparison.png
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

WANDB_DIR   = 'wandb_data'
PLOTS_DIR   = 'plots'
METRIC      = 'train/episode_reward'
SMOOTH      = 3       # rolling-window size for smoothing
ALPHA_SHADE = 0.25


def load_runs(task, exp_name):
    """Load all seed history CSVs for a (task, exp_name) pair. Returns list of DataFrames."""
    pattern = os.path.join(WANDB_DIR, f'{task}_{exp_name}_seed*_history.csv')
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files found for pattern: {pattern}')
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = df[['_step', METRIC, 'phase'] if 'phase' in pd.read_csv(f, nrows=0).columns else ['_step', METRIC]].copy()
        dfs.append(df)
    print(f'  {exp_name}: {len(files)} seeds')
    return dfs


def interpolate_to_common_grid(dfs, col, step_col='_step', n_points=200):
    """Interpolate each seed's series onto a common step grid; return (grid, matrix)."""
    all_steps = np.concatenate([df[step_col].dropna().values for df in dfs])
    grid = np.linspace(all_steps.min(), all_steps.max(), n_points)
    matrix = []
    for df in dfs:
        sub = df[[step_col, col]].dropna()
        interp = np.interp(grid, sub[step_col].values, sub[col].values)
        matrix.append(interp)
    return grid, np.array(matrix)


def smooth(arr, window):
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def plot_band(ax, steps, matrix, color, label, linestyle='-', smooth_w=SMOOTH):
    mean = np.nanmean(matrix, axis=0)
    std  = np.nanstd(matrix, axis=0)
    mean = smooth(mean, smooth_w)
    std  = smooth(std,  smooth_w)
    ax.plot(steps, mean, color=color, linestyle=linestyle, linewidth=1.8, label=label)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=ALPHA_SHADE)


def make_plot(task, tdmpc_exp, o2_exp, smooth_w, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    # --- TDMPC baseline (red solid) ---
    tdmpc_dfs = load_runs(task, tdmpc_exp)
    tdmpc_step, tdmpc_mat = interpolate_to_common_grid(tdmpc_dfs, METRIC)
    n_tdmpc = len(tdmpc_dfs)
    plot_band(ax, tdmpc_step, tdmpc_mat, color='red',
              label=f'TDMPC (N=512)', smooth_w=smooth_w)

    # --- O2 phased runs: split by phase ---
    o2_dfs = load_runs(task, o2_exp)
    n_o2 = len(o2_dfs)

    # TDMPC phase of O2 run (phase 1 or 2 → red dashed)
    o2_tdmpc_dfs = []
    o2_latent_dfs = []
    o2_transition_steps = []

    for df in o2_dfs:
        if 'phase' not in df.columns:
            print('  Warning: no phase column — treating whole run as TDMPC phase')
            o2_tdmpc_dfs.append(df)
            continue
        tdmpc_mask  = df['phase'].isin([1, 2])
        latent_mask = df['phase'] == 3

        o2_tdmpc_dfs.append(df[tdmpc_mask].copy())
        o2_latent_dfs.append(df[latent_mask].copy())

        if latent_mask.any():
            o2_transition_steps.append(df.loc[latent_mask, '_step'].min())

    # Red dashed: O2 TDMPC phase
    if o2_tdmpc_dfs and any(len(d) > 1 for d in o2_tdmpc_dfs):
        step_grid, mat = interpolate_to_common_grid(o2_tdmpc_dfs, METRIC)
        plot_band(ax, step_grid, mat, color='red',
                  label=f'O2 — TDMPC phase (N=512)', linestyle='--', smooth_w=smooth_w)

    # Blue solid: O2 latent phase
    if o2_latent_dfs and any(len(d) > 1 for d in o2_latent_dfs):
        step_grid, mat = interpolate_to_common_grid(o2_latent_dfs, METRIC)
        plot_band(ax, step_grid, mat, color='blue',
                  label=f'O2 — O2 phase (N=32)', linestyle='-', smooth_w=smooth_w)

    # Vertical line at O2 phase start
    if o2_transition_steps:
        transition = np.median(o2_transition_steps)
        ax.axvline(transition, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(transition + 500, ax.get_ylim()[1] * 0.95, 'O2 phase starts',
                fontsize=9, color='gray', va='top')

    task_title = task.replace('-', ' ').title()
    ax.set_title(f'{task_title} — TDMPC vs O2', fontsize=13)
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'\nSaved plot to {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',       required=True,          help='e.g. cartpole-swingup')
    parser.add_argument('--tdmpc-exp',  default='TDMPC_normal', help='exp_name for TDMPC baseline runs')
    parser.add_argument('--o2-exp',     default='GAN_tryNo2',   help='exp_name for O2 phased runs')
    parser.add_argument('--smooth',     type=int, default=SMOOTH, help='rolling average window size')
    parser.add_argument('--out',        default=None,           help='output path (default: plots/{task}_comparison.png)')
    args = parser.parse_args()

    out = args.out or os.path.join(PLOTS_DIR, f'{args.task.replace("-", "_")}_comparison.png')
    make_plot(args.task, args.tdmpc_exp, args.o2_exp, args.smooth, out)
