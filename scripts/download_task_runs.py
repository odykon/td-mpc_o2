"""
download_task_runs.py
---------------------
Downloads W&B runs for a given task and a list of exp_names, saving history + config CSVs.

Usage:
    python scripts/download_task_runs.py --task cartpole-swingup --exps TDMPC_normal GAN_tryNo2
    python scripts/download_task_runs.py --task walker-walk --exps TDMPC_normal o2_phased
"""

import os
import argparse
import pandas as pd
import wandb

ENTITY  = 'odysseaskon-national-technical-university-of-athens'
PROJECT = 'TDMPC_O2'
OUT_DIR = 'wandb_data'

os.makedirs(OUT_DIR, exist_ok=True)


def download_runs(task, exp_names, states=('finished',)):
    api  = wandb.Api()
    runs = api.runs(f'{ENTITY}/{PROJECT}', filters={'config.task': task})
    print(f'Found {len(runs)} runs for task={task}, filtering to exps={exp_names}, states={states}')

    saved = 0
    for run in runs:
        exp  = run.config.get('exp_name', '')
        seed = run.config.get('seed', '?')
        if exp not in exp_names:
            continue
        if run.state not in states:
            print(f'  Skipping {run.id} ({exp} seed{seed}) — state={run.state}')
            continue

        safe_name = f'{task}_{exp}_seed{seed}_{run.id}'
        hist_path = os.path.join(OUT_DIR, f'{safe_name}_history.csv')
        cfg_path  = os.path.join(OUT_DIR, f'{safe_name}_config.csv')

        history = run.history()
        history.to_csv(hist_path, index=False)

        config_df = pd.DataFrame([run.config])
        config_df.to_csv(cfg_path, index=False)

        print(f'  Saved: {safe_name}')
        saved += 1

    print(f'\nDone. {saved} runs saved to {OUT_DIR}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help='e.g. cartpole-swingup')
    parser.add_argument('--exps', nargs='+', required=True, help='exp_name values to include')
    parser.add_argument('--states', nargs='+', default=['finished'], help='run states to include')
    args = parser.parse_args()

    download_runs(args.task, set(args.exps), set(args.states))
