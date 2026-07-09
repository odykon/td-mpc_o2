"""
TD-MPC training script — CEM mean warm-starting disabled.
Identical to train_tdmpc.py, except every call to agent.plan() passes t0=True,
so tdmpc.py's `if not t0 and hasattr(self, '_prev_mean')` branch never fires and
the CEM mean is re-initialized to zero every step instead of carrying over the
previous step's converged mean.

Usage (from repo root):
    python scripts/train_tdmpc_no_warmstart.py task=walker-walk seed=1
    python scripts/train_tdmpc_no_warmstart.py task=cheetah-run seed=1

Logs are saved to logs/<task>/<modality>/<exp_name>/<seed>/
  - train.csv : per-episode training metrics
  - eval.csv  : periodic evaluation rewards
  - config.yaml: the full config used for this run
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tdmpc' / 'src'))

import torch
import numpy as np
import time

from omegaconf import OmegaConf
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer, linear_schedule
from o2.training_utils import set_seed, update_tdmpc
from o2.logger import CSVLogger

torch.backends.cudnn.benchmark = True

CFG_PATH = REPO_ROOT / 'tdmpc' / 'cfgs'
LOG_ROOT = REPO_ROOT / 'logs'

# Mirrors PHASED_DEFAULTS' step budget (train_o2_phased.py) so this baseline
# is directly comparable: same total MuJoCo steps and same horizon/std decay length.
NO_WARMSTART_DEFAULTS = {
    'mujoco_train_steps':           40000,
    'mujoco_seed_steps':            4000,
    'mujoco_std_schedule_steps':    40000,
    'mujoco_horizon_schedule_steps': 40000,
    'exp_name':                     'tdmpc_no_warmstart',
}


@torch.no_grad()
def evaluate(env, agent, num_episodes: int, step: int) -> float:
    """Run agent in eval mode and return mean episode reward."""
    rewards = []
    for _ in range(num_episodes):
        obs, done, total, t = env.reset(), False, 0.0, 0
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=True)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            total += reward
            t += 1
        rewards.append(total)
    return float(np.mean(rewards))


def train(cfg):
    """Main training loop for TD-MPC, without CEM mean warm-starting."""
    assert torch.cuda.is_available(), 'CUDA is required. Use a GPU runtime.'
    set_seed(cfg.seed)

    work_dir = LOG_ROOT / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    logger = CSVLogger(work_dir, cfg)

    env = make_env(cfg)
    agent = TDMPC(cfg)
    buffer = ReplayBuffer(cfg)

    print('=' * 60)
    print(OmegaConf.to_yaml(cfg))
    print('=' * 60)
    print(f'Task:        {cfg.task}')
    print(f'Train steps: {cfg.train_steps * cfg.action_repeat:,}  (env steps)')
    print(f'Obs shape:   {cfg.obs_shape}')
    print(f'Action dim:  {cfg.action_dim}')
    print(f'Seed:        {cfg.seed}')
    print(f'Log dir:     {work_dir}')
    print(f'Warm-start:  disabled (t0=True every step)')
    print('=' * 60 + '\n')

    episode_idx = 0
    start_time = time.time()

    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
        # --- Collect one episode ---
        t_ep = time.time()
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            action = agent.plan(obs, step=step, t0=True)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
        assert len(episode) == cfg.episode_length
        buffer += episode
        ep_time = time.time() - t_ep

        # --- Update TOLD model ---
        train_metrics = {}
        update_time = 0.0
        if step >= cfg.seed_steps:
            t_update = time.time()
            train_metrics = update_tdmpc(agent, buffer, step)
            update_time = time.time() - t_update

        # --- Log training episode ---
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        logger.log_train({
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward,
            'horizon': int(linear_schedule(cfg.horizon_schedule, step)),
            'std': linear_schedule(cfg.std_schedule, step),
            'ep_time': ep_time,
            'update_time': update_time,
            **train_metrics,
        })

        # --- Periodic evaluation ---
        if env_step % cfg.eval_freq == 0 and cfg.eval_episodes > 0:
            eval_reward = evaluate(env, agent, cfg.eval_episodes, step)
            logger.log_eval({
                'episode': episode_idx,
                'env_step': env_step,
                'episode_reward': eval_reward,
                'total_time': time.time() - start_time,
            })

        # --- Save model checkpoint ---
        if cfg.get('save_model', False) and env_step % cfg.eval_freq == 0 and env_step > 0:
            ckpt_dir = work_dir / 'models'
            ckpt_dir.mkdir(exist_ok=True)
            agent.save(ckpt_dir / f'model_{env_step}.pt')

    if cfg.get('save_model', False):
        agent.save(work_dir / 'final_model.pt')
        torch.save(buffer.__dict__, work_dir / 'replay_buffer.pth')
        print(f'Saved model and buffer to {work_dir}')

    logger.close()
    print('\nTraining complete.')


def make_cfg(task: str, **overrides) -> OmegaConf:
    """
    Build a config programmatically for use in notebooks.

    Example:
        from scripts.train_tdmpc_no_warmstart import make_cfg
        cfg = make_cfg('walker-walk', seed=1, exp_name='my_run')
        cfg.lr = 3e-4
        OmegaConf.save(cfg, 'my_cfg.yaml')
        # then: !python scripts/train_tdmpc_no_warmstart.py cfg=my_cfg.yaml
    """
    old_argv = sys.argv
    sys.argv = ['train_tdmpc_no_warmstart', f'task={task}'] + [f'{k}={v}' for k, v in overrides.items()]
    try:
        cfg = parse_cfg(CFG_PATH)
    finally:
        sys.argv = old_argv
    return cfg


def load_cfg() -> OmegaConf:
    """
    Load config with NO_WARMSTART_DEFAULTS and an optional custom YAML file.

    Priority (lowest to highest):
      1. tdmpc/cfgs/default.yaml
      2. tdmpc/cfgs/tasks/<domain>.yaml
      3. NO_WARMSTART_DEFAULTS (mirrored in cfgs/train_tdmpc_no_warmstart.yaml)
      4. Custom YAML passed as cfg=<path>
      5. Remaining CLI args (e.g. seed=1 exp_name=test)

    train_steps/seed_steps and horizon_schedule/std_schedule are all rebuilt
    from mujoco_* (raw MuJoCo step) counts divided by action_repeat — same
    approach as train_o2_phased.py's load_cfg — so this baseline's step budget
    and schedule length line up with the phased O2 runs.

    Example:
      python scripts/train_tdmpc_no_warmstart.py cfg=cfgs/train_tdmpc_no_warmstart.yaml task=walker-walk seed=1
    """
    cfg = parse_cfg(CFG_PATH)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(NO_WARMSTART_DEFAULTS))

    custom_path = cfg.get('cfg', None)
    if custom_path:
        custom = OmegaConf.load(custom_path)
        cli = OmegaConf.from_cli()
        cli_overrides = OmegaConf.create({k: v for k, v in cli.items() if k != 'cfg'})
        cfg = OmegaConf.merge(cfg, custom, cli_overrides)

    for k, v in cfg.items():
        if isinstance(v, str):
            match = re.match(r'^(\d+)([+\-*/])(\d+)$', v)
            if match:
                result = eval(match.group(1) + match.group(2) + match.group(3))
                cfg[k] = int(result) if isinstance(result, float) and result.is_integer() else result

    ar = cfg.action_repeat
    cfg.train_steps = int(cfg.mujoco_train_steps) // ar
    cfg.seed_steps = int(cfg.mujoco_seed_steps) // ar

    std_steps = int(cfg.mujoco_std_schedule_steps) // ar
    horizon_steps = int(cfg.mujoco_horizon_schedule_steps) // ar
    cfg.std_schedule = f"linear(0.5, {cfg.min_std}, {std_steps})"
    cfg.horizon_schedule = f"linear(1, {cfg.horizon}, {horizon_steps})"

    return cfg


if __name__ == '__main__':
    cfg = load_cfg()
    train(cfg)
