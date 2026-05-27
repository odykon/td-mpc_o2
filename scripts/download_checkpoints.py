"""
download_checkpoints.py — Download intermediate model + buffer artifacts from W&B.

Downloads to:
    checkpoints/<task>/<seed>/model.pt
    checkpoints/<task>/<seed>/buffer.pth

Usage:
    python scripts/download_checkpoints.py
"""

import os
import sys
import glob
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import wandb

ENTITY  = 'odysseaskon-national-technical-university-of-athens'
PROJECT = 'TDMPC_O2'

TASKS = [
    'cheetah-run',
    'walker-walk',
    'hopper-hop',
    'humanoid-walk',
    'cartpole-swingup',
]
SEEDS = [1, 2]

OUT_DIR = REPO_ROOT / 'checkpoints'

api = wandb.Api()

for task in TASKS:
    task_safe = task.replace('-', '_')
    for seed in SEEDS:
        dest = OUT_DIR / task / str(seed)
        dest.mkdir(parents=True, exist_ok=True)

        for kind, suffix, dest_name in [
            ('model',  'intermediate',        'model.pt'),
            ('buffer', 'intermediate_buffer', 'buffer.pth'),
        ]:
            artifact_name = f"{ENTITY}/{PROJECT}/{suffix}_{task_safe}_seed{seed}:latest"
            dest_file = dest / dest_name

            if dest_file.exists():
                print(f'[skip]  {artifact_name}  (already downloaded)')
                continue

            print(f'[fetch] {artifact_name}')
            try:
                art = api.artifact(artifact_name)
                tmp = str(dest) + f'/_tmp_{kind}'
                art.download(root=tmp)
                exts = {'model.pt': '*.pt', 'buffer.pth': '*.pth'}
                files = glob.glob(os.path.join(tmp, exts[dest_name]))
                if not files:
                    print(f'  WARNING: no {dest_name} found in artifact, skipping.')
                else:
                    shutil.move(files[0], dest_file)
                    print(f'  -> {dest_file}')
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception as e:
                print(f'  ERROR: {e}')

print('\nDone.')
