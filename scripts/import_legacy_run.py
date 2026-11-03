"""
Copy a legacy TD-MPC run (logs/<task>/<modality>/<exp_name>/<seed>/ layout) into experiments/.

    python scripts/import_legacy_run.py <src_dir> <algo> <task> <exp> <variant> <seed> --notes "..."

Copies config.yaml, train.csv, eval.csv, final_model.pt -> model_final.pt,
replay_buffer.pth -> buffer.pth, and writes meta.json. Intermediate checkpoints are not copied.
"""

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_utils import run_dir_path

FILES = {'config.yaml': 'config.yaml', 'train.csv': 'train.csv', 'eval.csv': 'eval.csv',
         'final_model.pt': 'model_final.pt', 'replay_buffer.pth': 'buffer.pth'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src_dir', type=Path)
    for a in ['algo', 'task', 'exp', 'variant']:
        p.add_argument(a)
    p.add_argument('seed', type=int)
    p.add_argument('--notes', default='imported from Kaggle legacy layout')
    p.add_argument('--host', default='kaggle')
    p.add_argument('--branch', default='known-dynamics-cartpole')
    args = p.parse_args()

    missing = [f for f in FILES if not (args.src_dir / f).exists()]
    if missing:
        sys.exit(f'missing in {args.src_dir}: {missing}')

    dst = run_dir_path(args.algo, args.task, args.exp, args.variant, args.seed)
    dst.mkdir(parents=True, exist_ok=False)
    for src_name, dst_name in FILES.items():
        shutil.copy2(args.src_dir / src_name, dst / dst_name)

    cfg = OmegaConf.load(args.src_dir / 'config.yaml')
    mtime = (args.src_dir / 'final_model.pt').stat().st_mtime
    meta = {
        'commit': 'unknown',
        'branch': args.branch,
        'date': datetime.datetime.fromtimestamp(mtime).isoformat(timespec='seconds'),
        'host': args.host,
        'cmd': (f'scripts/train_tdmpc.py cfg=cfgs/train_tdmpc.yaml task={cfg.task} seed={cfg.seed} '
                f'action_repeat={cfg.action_repeat} train_steps={cfg.train_steps} '
                f'exp_name={cfg.exp_name} save_model=true'),
        'notes': args.notes,
    }
    (dst / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'{args.src_dir} -> {dst}')


if __name__ == '__main__':
    main()
