"""
plot_scales.py
--------------
Value, reward, LML and decoder gradient scales per action repeat, averaged per
episode from each run's decoder_metrics.csv (episode 20 includes pretraining).

Usage (from repo root):  python o2_base/plot_scales.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = Path(__file__).resolve().parents[1] / 'runs' / 'o2_base'
TASK, SEED, TAG = 'cheetah-run', 1, 'monitor'
METRICS = ['buffer_reward', 'reward_part', 'q_part',
           'value_std_first', 'eff_elites_first', 'eff_elites_last',
           'grad_direct', 'grad_unroll', 'grad_u_mean']
LOG_SCALE = ['grad_direct', 'grad_unroll', 'grad_u_mean']


def per_episode(ar, metric):
    with open(RUNS / f'{TASK}_ar{ar}_s{SEED}_{TAG}' / 'decoder_metrics.csv') as f:
        rows = list(csv.DictReader(f))
    episodes = sorted({int(r['ep']) for r in rows})
    means = []
    for ep in episodes:
        values = [float(r[metric]) for r in rows if int(r['ep']) == ep]
        means.append(sum(values) / len(values))
    return episodes, means


fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for ax, metric in zip(axes.flat, METRICS):
    for ar in [1, 2, 3, 4]:
        ax.plot(*per_episode(ar, metric), label=f'ar {ar}')
    if metric in LOG_SCALE:
        ax.set_yscale('log')
    ax.set_title(metric)
    ax.set_xlabel('episode')
    ax.legend()

fig.tight_layout()
fig.savefig(RUNS / 'scales.png')
print(f'saved {RUNS / "scales.png"}')
