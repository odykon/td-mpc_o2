"""
TD-MPC + O2 training script, kept as close as possible to the vendored
tdmpc/src/train.py (same loop structure, same wandb Logger, same key names) —
the only additions are the latent decoder, its update step, and swapping
agent.plan() for TDMPC_O2's CEM_in_latent() during rollout/eval.

Does not modify or add anything under tdmpc/ — only imports its existing
modules (cfg, env, algorithm.helper, logger) via sys.path, same pattern as
scripts/train_tdmpc.py. Logs go to <repo-root>/logs/, not tdmpc/logs/, so this
can be run directly from the repo root with no `cd` into tdmpc/ required.

Usage (from repo root):
    python scripts/train_o2_vanilla.py task=walker-walk seed=1
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

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
from algorithm.helper import Episode, ReplayBuffer
from o2.tdmpc_o2 import TDMPC_O2
from o2.training_utils import set_seed, update_decoder
import logger

torch.backends.cudnn.benchmark = True

CFG_PATH = REPO_ROOT / 'tdmpc' / 'cfgs'
O2_DEFAULTS_PATH = REPO_ROOT / 'cfgs' / 'o2_default.yaml'
LOG_ROOT = REPO_ROOT / 'logs'


def evaluate(env, agent, num_episodes, step, env_step, video):
    """Same as tdmpc/src/train.py's evaluate(), but planning via CEM_in_latent
    instead of TDMPC.plan(). Uses sample_final_action=False (the CEM search
    distribution's mean, deterministic) here, unlike the training rollout in
    train() below which samples — closer to tdmpc's own eval_mode, which drops
    the training-time action noise for evaluation."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video: video.init(env, enabled=(i == 0))
        while not done:
            action, *_ = agent.CEM_in_latent(obs, step=step, sample_final_action=False, t0=(t == 0))
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            if video: video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        if video: video.save(env_step)
    return np.nanmean(episode_rewards)


def train(cfg):
    """Training script for TD-MPC + O2 latent decoder. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = LOG_ROOT / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env = make_env(cfg)
    cfg.latent_action_dim = cfg.horizon * cfg.action_dim
    agent, buffer = TDMPC_O2(cfg), ReplayBuffer(cfg)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):

        # Collect trajectory
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            # TDMPC.plan() has a built-in `step < seed_steps` random-action branch;
            # CEM_in_latent has no equivalent, so it's replicated here explicitly.
            if step < cfg.seed_steps:
                action = torch.empty(cfg.action_dim, dtype=torch.float32, device=agent.device).uniform_(-1, 1)
            else:
                action, *_ = agent.CEM_in_latent(obs, step=step, sample_final_action=True, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
        assert len(episode) == cfg.episode_length
        buffer += episode

        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i))

            # --- added: decoder update (no equivalent in the original TD-MPC) ---
            # Gated to every cfg.decoder_update_interval-th episode (default 1 = every episode).
            if episode_idx % cfg.decoder_update_interval == 0:
                dec_metrics = update_decoder(agent, buffer, cfg, step)
                dec_metrics.pop('grad_tracker', None)
                train_metrics.update({f'decoder_{k}': v for k, v in dec_metrics.items()})

        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward}
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train')

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
            L.log(common_metrics, category='eval')

        # Save periodic model checkpoint (optional, off unless cfg.save_model is set)
        if cfg.get('save_model', False) and env_step % cfg.eval_freq == 0 and env_step > 0:
            ckpt_dir = work_dir / 'models'
            ckpt_dir.mkdir(exist_ok=True)
            agent.save(ckpt_dir / f'model_{env_step}.pt')

    L.finish(agent)
    if cfg.get('save_model', False):
        torch.save(buffer.__dict__, work_dir / 'replay_buffer.pth')
        print(f'Saved replay buffer to {work_dir}')
    print('Training completed successfully')


def load_cfg() -> OmegaConf:
    """Same as tdmpc/src/cfg.py's parse_cfg(), plus cfgs/o2_default.yaml merged
    in underneath any real CLI overrides — TDMPC_O2 needs its decoder/CEM keys
    (latent_num_samples, flow_num_layers, told_updates, decoder_updates, ...)
    that plain tdmpc/cfgs/default.yaml doesn't define."""
    cfg = parse_cfg(CFG_PATH)
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, OmegaConf.load(O2_DEFAULTS_PATH), cli)
    return cfg


if __name__ == '__main__':
    train(load_cfg())
