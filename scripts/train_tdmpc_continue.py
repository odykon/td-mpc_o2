"""
train_tdmpc_continue.py — Continue standard CEM training from a Phase 1 checkpoint.

Loads the model and buffer saved by train_o2_phased.py at the Phase 1 → Phase 2
transition, then runs standard CEM planning for the same Phase 2 duration.

W&B env_step starts from mujoco_latent_start_steps so curves overlay directly
with the corresponding o2_phased run when both are grouped by task.

Usage:
    python scripts/train_tdmpc_continue.py cfg=cfgs/exp_phased.yaml task=walker-walk seed=1
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import re
import sys
import glob
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tdmpc' / 'src'))

import torch
import time
import wandb

from omegaconf import OmegaConf
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer, linear_schedule
from o2.training_utils import set_seed, update_tdmpc
from o2.eval_utils import evaluate_agent

torch.backends.cudnn.benchmark = True

CFG_PATH = REPO_ROOT / 'tdmpc' / 'cfgs'


def train(cfg):
    assert torch.cuda.is_available(), 'CUDA is required.'
    set_seed(cfg.seed)

    task_safe   = cfg.task.replace('-', '_')
    model_path  = os.path.join(cfg.checkpoint_dir, f'{task_safe}_seed{cfg.seed}_phase1.pt')
    buffer_path = os.path.join(cfg.checkpoint_dir, f'{task_safe}_seed{cfg.seed}_phase1_buffer.pth')

    # Fail early and clearly if the phase 1 checkpoint is missing,
    # rather than training from scratch silently.
    assert os.path.exists(model_path),  f'Phase 1 model not found:  {model_path}'
    assert os.path.exists(buffer_path), f'Phase 1 buffer not found: {buffer_path}'

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.task,
            name=f"tdmpc_continue__seed{cfg.seed}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    env    = make_env(cfg)
    # Plain TDMPC — no decoder, no latent action space.
    # strict=False tolerates the decoder keys present in the TDMPC_O2 state dict.
    agent  = TDMPC(cfg)
    buffer = ReplayBuffer(cfg)

    state = torch.load(model_path, map_location=agent.device)
    agent.model.load_state_dict(state, strict=False)
    print(f'Loaded Phase 1 model:  {model_path}')

    # Restore the full buffer state (priorities, indices, tensors) so
    # the TOLD update distribution is identical to the o2_phased run.
    buffer_state = torch.load(buffer_path, map_location=buffer.device)
    buffer.__dict__.update(buffer_state)
    print(f'Loaded Phase 1 buffer: {buffer_path}')

    phase2_mujoco = cfg.mujoco_train_steps - cfg.mujoco_latent_start_steps
    print('=' * 60)
    print(f'Task:                {cfg.task}')
    print(f'Phase 2 MuJoCo steps:{phase2_mujoco:,}')
    print(f'Seed:                {cfg.seed}')
    print('=' * 60 + '\n')

    episode_idx = 0
    start_time  = time.time()

    def row(label, val):
        print(f'  {label:<22}: {val}')

    # Start the loop from latent_start_steps (not 0) so that:
    # 1. env_step = step * action_repeat naturally starts from mujoco_latent_start_steps
    # 2. std/horizon schedules continue from where Phase 1 left off
    for step in range(cfg.latent_start_steps,
                      cfg.train_steps + cfg.episode_length,
                      cfg.episode_length):

        # Collect episode with standard CEM throughout
        t_ep = time.time()
        obs  = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            action = agent.plan(obs, step=step, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
        buffer += episode
        episode_idx += 1
        ep_time = time.time() - t_ep

        # TOLD update — buffer is already populated from Phase 1 so no seed_steps needed
        t_update     = time.time()
        train_metrics = update_tdmpc(agent, buffer, step)
        update_time  = time.time() - t_update

        env_step   = int(step * cfg.action_repeat)
        horizon    = int(linear_schedule(cfg.horizon_schedule, step))
        std        = linear_schedule(cfg.std_schedule, step)
        total_time = time.time() - start_time

        SEP = '─' * 42
        print(f'\n{SEP}')
        print(f'  Episode {episode_idx}   step {env_step:,}   [tdmpc_continue]')
        print(SEP)
        row('Reward',      f'{episode.cumulative_reward:>10.1f}')
        row('Horizon',     f'{horizon:>10d}')
        row('Std',         f'{std:>10.3f}')
        row('Ep time',     f'{ep_time:>9.1f}s')
        row('Update time', f'{update_time:>9.1f}s')
        row('Total time',  f'{total_time:>9.0f}s')
        for k, v in train_metrics.items():
            row(k, f'{v:>10.4f}')

        if cfg.use_wandb:
            wandb.log({
                'episode':              episode_idx,
                'train/episode_reward': episode.cumulative_reward,
                'train/horizon':        horizon,
                'train/std':            std,
                **{f'train/{k}': v for k, v in train_metrics.items()},
            }, step=env_step)

    # Final eval using standard CEM (use_latent=False)
    total_env_step = int(cfg.train_steps * cfg.action_repeat)
    eval_tmp = tempfile.mkdtemp()
    try:
        eval_metrics = evaluate_agent(
            env, agent, cfg,
            step=total_env_step,
            n_episodes=cfg.eval_episodes,
            save_dir=eval_tmp,
            video_mode='first',
            use_latent=False,
        )
        if cfg.use_wandb:
            eval_log = {
                'eval/mean_reward': eval_metrics['mean_reward'],
                'eval/std_reward':  eval_metrics['std_reward'],
            }
            videos = glob.glob(os.path.join(eval_tmp, 'videos', '*.mp4'))
            if videos:
                eval_log['eval/video'] = wandb.Video(videos[0], fps=30, format='mp4')
            wandb.log(eval_log, step=total_env_step)
            wandb.finish()
    finally:
        shutil.rmtree(eval_tmp, ignore_errors=True)

    # Delete the checkpoint files — they were only needed for this handoff
    os.remove(model_path)
    os.remove(buffer_path)
    print(f'\nDone. Total time: {(time.time() - start_time) / 60:.1f} min')


def load_cfg() -> OmegaConf:
    """Same cfg loading as train_o2_phased — reused verbatim."""
    cfg = parse_cfg(CFG_PATH)

    # Import PHASED_DEFAULTS from the phased script to stay in sync
    from train_o2_phased import PHASED_DEFAULTS
    cfg = OmegaConf.merge(OmegaConf.create(PHASED_DEFAULTS), cfg)

    custom_path = cfg.get('cfg', None)
    if custom_path:
        custom = OmegaConf.load(custom_path)
        cli    = OmegaConf.from_cli()
        cli_overrides = OmegaConf.create({k: v for k, v in cli.items() if k != 'cfg'})
        cfg = OmegaConf.merge(cfg, custom, cli_overrides)

    for k, v in cfg.items():
        if isinstance(v, str):
            m = re.match(r'^(\d+)([+\-*/])(\d+)$', v)
            if m:
                result = eval(m.group(1) + m.group(2) + m.group(3))
                cfg[k] = int(result) if isinstance(result, float) and result.is_integer() else result

    ar = cfg.action_repeat
    cfg.train_steps         = int(cfg.mujoco_train_steps)         // ar
    cfg.seed_steps          = int(cfg.mujoco_seed_steps)          // ar
    cfg.decoder_start_steps = int(cfg.mujoco_decoder_start_steps) // ar
    cfg.latent_start_steps  = int(cfg.mujoco_latent_start_steps)  // ar

    std_steps     = int(cfg.mujoco_std_schedule_steps)     // ar
    horizon_steps = int(cfg.mujoco_horizon_schedule_steps) // ar
    cfg.std_schedule     = f"linear(0.5, {cfg.min_std}, {std_steps})"
    cfg.horizon_schedule = f"linear(1, {cfg.horizon}, {horizon_steps})"

    return cfg


if __name__ == '__main__':
    cfg = load_cfg()
    train(cfg)
