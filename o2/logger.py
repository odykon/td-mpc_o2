"""
logger.py
---------
CSV logger shared by all training scripts.

Writes train.csv and eval.csv to the run directory and prints a
formatted summary to the console after each episode.

O2-specific fields (phase, decoder_loss, decoder_time) are included
in the schema; they are simply left blank when logging a standard
TDMPC run.
"""

import csv
from pathlib import Path

from omegaconf import OmegaConf


class CSVLogger:
    TRAIN_FIELDS = [
        'episode', 'step', 'env_step', 'total_time', 'episode_reward',
        'horizon', 'std', 'ep_time', 'update_time',
        'phase', 'decoder_time', 'decoder_loss', 'decoder_grad_norm', 'saturation',
        'saturation_first',
        'consistency_loss', 'reward_loss', 'value_loss',
        'pi_loss', 'total_loss', 'weighted_loss', 'grad_norm',
    ]
    EVAL_FIELDS = ['episode', 'env_step', 'episode_reward', 'total_time']

    def __init__(self, log_dir: Path, cfg):
        log_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, log_dir / 'config.yaml')

        self._train_f = open(log_dir / 'train.csv', 'w', newline='')
        self._eval_f  = open(log_dir / 'eval.csv',  'w', newline='')
        self._train_w = csv.DictWriter(self._train_f, fieldnames=self.TRAIN_FIELDS, extrasaction='ignore')
        self._eval_w  = csv.DictWriter(self._eval_f,  fieldnames=self.EVAL_FIELDS,  extrasaction='ignore')
        self._train_w.writeheader()
        self._eval_w.writeheader()

        print(f'Logging to {log_dir}')

    def log_train(self, d: dict):
        row = {k: d.get(k, '') for k in self.TRAIN_FIELDS}
        self._train_w.writerow(row)
        self._train_f.flush()

        phase = d.get('phase', 'tdmpc')
        W = 38
        print('─' * W)
        print(f'  Episode {d["episode"]}   step {d["env_step"]:,}   [{phase}]')
        print('─' * W)
        print(f'  {"Reward":<16}: {d.get("episode_reward", 0):>8.1f}')
        print(f'  {"Horizon":<16}: {d.get("horizon", ""):>8}')
        print(f'  {"Std":<16}: {d.get("std", 0):>8.3f}')
        print(f'  {"Ep time":<16}: {d.get("ep_time", 0):>7.1f}s')
        print(f'  {"Update time":<16}: {d.get("update_time", 0):>7.1f}s')
        if phase == 'o2':
            print(f'  {"Decoder time":<16}: {d.get("decoder_time", 0):>7.1f}s')
            print(f'  {"Decoder loss":<16}: {d.get("decoder_loss", 0):>8.4f}')
            print(f'  {"Decoder grad":<16}: {d.get("decoder_grad_norm", 0):>8.4f}')
        print(f'  {"Total time":<16}: {d.get("total_time", 0):>7.0f}s')
        if d.get('total_loss', '') != '':
            print(f'  {"total_loss":<16}: {d["total_loss"]:>8.3f}')
            print(f'  {"reward_loss":<16}: {d["reward_loss"]:>8.3f}')
            print(f'  {"value_loss":<16}: {d["value_loss"]:>8.3f}')
            print(f'  {"pi_loss":<16}: {d["pi_loss"]:>8.3f}')

    def log_eval(self, d: dict):
        row = {k: d.get(k, '') for k in self.EVAL_FIELDS}
        self._eval_w.writerow(row)
        self._eval_f.flush()

        W = 38
        print('═' * W)
        print(f'  EVAL   episode {d["episode"]}   step {d["env_step"]:,}')
        print(f'  {"Reward":<16}: {d.get("episode_reward", 0):>8.1f}')
        print('═' * W)

    def close(self):
        self._train_f.close()
        self._eval_f.close()
