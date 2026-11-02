"""
plot_sweep.py
-------------
O2 vs TD-MPC-only episode reward, one panel per action repeat.

Usage (from repo root):
    python o2_base/plot_sweep.py           # O2 runs from the first sweep -> comparison.png
    python o2_base/plot_sweep.py monitor   # O2 runs tagged _monitor     -> comparison_monitor.png
The TD-MPC baseline is always the untagged _tdmpc runs.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = Path(__file__).resolve().parents[1] / 'runs' / 'o2_base'
TASK, SEED, PHASE1_EPISODES = 'cheetah-run', 1, 20
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
O2_SUFFIX = f'_{TAG}' if TAG else ''
OUT = RUNS / f'comparison{O2_SUFFIX}.png'


def rewards(name):
    with open(RUNS / name / 'metrics.csv') as f:
        return [float(row['reward']) for row in csv.DictReader(f)]


fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, ar in zip(axes.flat, [1, 2, 3, 4]):
    ax.plot(rewards(f'{TASK}_ar{ar}_s{SEED}{O2_SUFFIX}'), label='O2')
    ax.plot(rewards(f'{TASK}_ar{ar}_s{SEED}_tdmpc'), label='TD-MPC')
    ax.axvline(PHASE1_EPISODES, color='k', linestyle='--')
    ax.set_title(f'action repeat {ar}')
    ax.set_xlabel('episode')
    ax.set_ylabel('episode reward')
    ax.legend()

fig.tight_layout()
fig.savefig(OUT)
print(f'saved {OUT}')
