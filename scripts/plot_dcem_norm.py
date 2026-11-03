"""
Plots for scripts/compare_dcem_norm.py.

    python scripts/plot_dcem_norm.py experiments/o2_base/cheetah-run/<exp>

Writes to <exp_dir>/plots/:
    returns.png     rows = AR, cols = seed: eval return vs decoder updates per mode,
                    TD-MPC reference dashed
    eff_elites.png  rows = AR, cols = (iter 0, last iter): median eff_elites vs update
                    per mode, mean over seeds
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def load_runs(exp_dir):
    runs = []
    for d in sorted(exp_dir.iterdir()):
        m = re.fullmatch(r'ar(\d+)_std-(.+)_s(\d+)', d.name)
        if m and (d / 'eval.csv').exists():
            runs.append({'ar': int(m[1]), 'mode': m[2], 'seed': int(m[3]), 'dir': d})
    return runs


def main():
    exp_dir = Path(sys.argv[1])
    out = exp_dir / 'plots'
    out.mkdir(exist_ok=True)
    runs = load_runs(exp_dir)
    ars = sorted({r['ar'] for r in runs})
    seeds = sorted({r['seed'] for r in runs})
    modes = sorted({r['mode'] for r in runs})
    colors = {m: f'C{i}' for i, m in enumerate(modes)}
    ref_path = exp_dir / 'reference_tdmpc.csv'
    ref = pd.read_csv(ref_path) if ref_path.exists() else None

    # returns
    fig, axes = plt.subplots(len(ars), len(seeds), figsize=(4 * len(seeds), 3 * len(ars)),
                             squeeze=False, sharex=True)
    for i, ar in enumerate(ars):
        for j, seed in enumerate(seeds):
            ax = axes[i, j]
            for r in runs:
                if r['ar'] == ar and r['seed'] == seed:
                    ev = pd.read_csv(r['dir'] / 'eval.csv')
                    ax.plot(ev['n_updates'], ev['episode_return'], 'o-', color=colors[r['mode']], label=r['mode'])
            if ref is not None:
                row = ref[(ref['ar'] == ar) & (ref['seed'] == seed)]
                if len(row):
                    ax.axhline(row['tdmpc_return'].iloc[0], color='k', ls='--', label='TD-MPC')
            ax.set_title(f'AR {ar}, seed {seed}')
            if j == 0:
                ax.set_ylabel('episode return')
            if i == len(ars) - 1:
                ax.set_xlabel('decoder updates')
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / 'returns.png', dpi=120)

    # eff_elites at iteration 0 and last
    fig, axes = plt.subplots(len(ars), 2, figsize=(9, 3 * len(ars)), squeeze=False, sharex=True)
    for i, ar in enumerate(ars):
        for mode in modes:
            it = pd.concat([pd.read_csv(r['dir'] / 'iters.csv') for r in runs
                            if r['ar'] == ar and r['mode'] == mode])
            it = it[it['update'] >= 0]
            last = it['iter'].max()
            for j, k in enumerate([0, last]):
                s = it[it['iter'] == k].groupby('update')['eff_elites_med'].mean()
                axes[i, j].plot(s.index, s.values, '.-', color=colors[mode], label=mode)
        for j, name in enumerate(['iter 0', 'last iter']):
            axes[i, j].set_title(f'AR {ar}, {name}')
            axes[i, j].set_ylabel('eff_elites (median over batch)')
        axes[i, 0].axhline(8, color='gray', ls=':')
        axes[i, 1].axhline(8, color='gray', ls=':')
    for j in range(2):
        axes[-1, j].set_xlabel('decoder update')
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / 'eff_elites.png', dpi=120)
    print(f'saved plots to {out}')


if __name__ == '__main__':
    main()
