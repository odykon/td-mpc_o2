"""
experiments_utils.py
--------------------
Run directory convention (see experiments/README.md):

    experiments/<algo>/<task>/<YYYY-MM-DD_slug>/<variant>_s<seed>/
        config.yaml   resolved cfg
        meta.json     commit, dirty, date, host, cmd, notes
"""

import datetime
import json
import socket
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

EXPERIMENTS_ROOT = Path(__file__).resolve().parent / 'experiments'


def run_dir_path(algo, task, exp, variant, seed, root=EXPERIMENTS_ROOT):
    name = f'{variant}_s{seed}' if variant else f's{seed}'
    return Path(root) / algo / task / exp / name


def git_meta():
    """commit hash and whether tracked files differ from it (untracked files are ignored)."""
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True,
                                    cwd=Path(__file__).resolve().parent).stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'),
            'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}


def write_meta(run_dir, notes=None, **fields):
    meta = {**git_meta(),
            'date': datetime.datetime.now().isoformat(timespec='seconds'),
            'host': socket.gethostname(),
            'cmd': ' '.join(sys.argv),
            **fields}
    if notes:
        meta['notes'] = notes
    (Path(run_dir) / 'meta.json').write_text(json.dumps(meta, indent=2))


def make_run_dir(algo, task, exp, variant, seed, cfg, root=EXPERIMENTS_ROOT, notes=None):
    """Create a new run directory with config.yaml and meta.json. Never overwrites an existing run."""
    run_dir = run_dir_path(algo, task, exp, variant, seed, root)
    run_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, run_dir / 'config.yaml')
    write_meta(run_dir, notes=notes)
    return run_dir
