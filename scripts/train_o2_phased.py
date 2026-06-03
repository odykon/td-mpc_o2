"""
train_o2_phased.py — Single-loop phased training with warm-started decoder.

One continuous training loop (one buffer) divided into three phases:

  tdmpc   (0 → mujoco_decoder_start_steps):
      Standard CEM planning, TOLD-only updates (500/ep).

  warmup  (mujoco_decoder_start_steps → mujoco_latent_start_steps):
      Standard CEM planning, TOLD (500/ep) + decoder (100/ep).
      Warm-starts the decoder before latent CEM begins.

  o2      (mujoco_latent_start_steps → mujoco_train_steps):
      Latent CEM planning, TOLD (500/ep) + decoder (100/ep).

All step thresholds are in raw MuJoCo interactions and divided by
action_repeat in load_cfg. One W&B run per (task, seed). Nothing is
persisted locally — all artifacts (models, buffer, video) are uploaded
directly to W&B and temp files are deleted immediately after.

Usage:
    python scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=walker-walk seed=1
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
from algorithm.helper import Episode, ReplayBuffer, linear_schedule
from o2.tdmpc_o2 import TDMPC_O2
from o2.training_utils import set_seed, update_tdmpc, update_decoder
from o2.eval_utils import evaluate_agent

torch.backends.cudnn.benchmark = True

CFG_PATH = REPO_ROOT / 'tdmpc' / 'cfgs'

PHASED_DEFAULTS = {
    # MuJoCo step thresholds (divided by action_repeat in load_cfg)
    'mujoco_train_steps':          40000,
    'mujoco_seed_steps':           4000,   # random exploration
    'mujoco_decoder_start_steps':  15000,  # decoder warm-up starts
    'mujoco_latent_start_steps':   20000,  # latent CEM starts

    # Schedule endpoints in MuJoCo steps (divided by action_repeat in load_cfg)
    # std decays from 0.5 → min_std; horizon grows from 1 → horizon
    'mujoco_std_schedule_steps':     40000,
    'mujoco_horizon_schedule_steps': 40000,

    # Update cadence
    'told_updates':     500,   # TOLD updates per episode (always, after seed)
    'decoder_updates':  100,   # decoder updates per episode (warmup + o2)

    # O2 architecture
    'latent_action_dim':  128,
    'decoder_init':       False,
    'use_latent_state':   True,
    'dcem_batch_size':    64,
    'latent_num_samples': 32,
    'latent_num_elites':  8,
    'lml_temperature':    1,
    'use_is_weights':     True,
    'dec_grad_clip_norm': 10.0,
    'use_wandb':          True,

    # Eval (one at end with video)
    'eval_episodes': 1,

    # W&B
    'wandb_project': 'TDMPC_O2',
    'wandb_entity':  'odysseaskon-national-technical-university-of-athens',
    'exp_name':      'o2_phased',

    # Required by TDMPC_O2 init; overwritten in load_cfg
    'decoder_start_steps': 0,
    'latent_start_steps':  0,
}



def train(cfg):
    assert torch.cuda.is_available(), 'CUDA is required.'
    set_seed(cfg.seed)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.task,
            name=f"{cfg.exp_name}__seed{cfg.seed}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    env    = make_env(cfg)
    agent  = TDMPC_O2(cfg)
    buffer = ReplayBuffer(cfg)

    print('=' * 60)
    print(f'Task:                {cfg.task}')
    print(f'Total MuJoCo steps:  {cfg.mujoco_train_steps:,}')
    print(f'  Seed ends at:      {cfg.mujoco_seed_steps:,}  MuJoCo')
    print(f'  Decoder warm-up:   {cfg.mujoco_decoder_start_steps:,}  MuJoCo')
    print(f'  Latent CEM starts: {cfg.mujoco_latent_start_steps:,}  MuJoCo')
    print(f'TOLD updates/ep:     {cfg.told_updates}')
    print(f'Decoder updates/ep:  {cfg.decoder_updates}')
    print(f'Seed:                {cfg.seed}')
    print('=' * 60 + '\n')

    episode_idx = 0
    start_time  = time.time()

    def row(label, val):
        print(f'  {label:<22}: {val}')

    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
        # Determine phase
        if step < cfg.decoder_start_steps:
            phase = 'tdmpc'
        elif step < cfg.latent_start_steps:
            phase = 'warmup'
        else:
            phase = 'o2'

        # Collect episode
        t_ep = time.time()
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            if step < cfg.seed_steps:
                action_np = env.action_space.sample()
                action = torch.tensor(action_np, dtype=torch.float32, device=agent.device)
            elif phase in ('tdmpc', 'warmup'):
                action = agent.plan(obs, step=step, t0=episode.first)
            else:
                action, *_ = agent.CEM_in_latent(
                    obs, step=step, sample_final_action=True
                )
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
        buffer += episode
        episode_idx += 1
        ep_time = time.time() - t_ep

        # Updates (gated on seed_steps)
        train_metrics = {}
        dec_metrics   = {}
        update_time   = 0.0
        decoder_time  = 0.0
        if step >= cfg.seed_steps:
            t_update = time.time()
            train_metrics = update_tdmpc(agent, buffer, step)
            update_time = time.time() - t_update
            if phase in ('warmup', 'o2'):
                t_dec = time.time()
                dec_metrics = update_decoder(agent, buffer, cfg, step)
                decoder_time = time.time() - t_dec

        env_step   = int(step * cfg.action_repeat)
        phase_code = {'tdmpc': 1, 'warmup': 2, 'o2': 3}[phase]
        horizon    = int(linear_schedule(cfg.horizon_schedule, step))
        std        = linear_schedule(cfg.std_schedule, step)
        total_time = time.time() - start_time

        SEP = '─' * 42
        print(f'\n{SEP}')
        print(f'  Episode {episode_idx}   step {env_step:,}   [{phase}]')
        print(SEP)
        row('Reward',       f'{episode.cumulative_reward:>10.1f}')
        row('Horizon',      f'{horizon:>10d}')
        row('Std',          f'{std:>10.3f}')
        row('Ep time',      f'{ep_time:>9.1f}s')
        if update_time:
            row('Update time',  f'{update_time:>9.1f}s')
        if decoder_time:
            row('Decoder time', f'{decoder_time:>9.1f}s')
        grad_tracker = dec_metrics.pop('grad_tracker', [])
        for k, v in dec_metrics.items():
            row(k, f'{v:>10.4f}')
        row('Total time',   f'{total_time:>9.0f}s')
        for k, v in train_metrics.items():
            row(k, f'{v:>10.4f}')

        if cfg.use_wandb:
            wandb.log({
                'phase':                phase_code,
                'episode':              episode_idx,
                'train/episode_reward': episode.cumulative_reward,
                'train/horizon':        horizon,
                'train/std':            std,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'decoder/{k}': v for k, v in dec_metrics.items()},
                **{f'decoder/dcem_iter_{i}_grad': g for i, g in grad_tracker},
            }, step=env_step)

    # ── Final evaluation with video ──────────────────────────────────────────
    total_step     = cfg.train_steps
    total_env_step = int(total_step * cfg.action_repeat)
    eval_tmp       = tempfile.mkdtemp()
    try:
        eval_metrics = evaluate_agent(
            env, agent, cfg,
            step=total_env_step,
            n_episodes=cfg.eval_episodes,
            save_dir=eval_tmp,
            video_mode='first',
            use_latent=(phase == 'o2'),
        )
        eval_log = {
            'eval/mean_reward': eval_metrics['mean_reward'],
            'eval/std_reward':  eval_metrics['std_reward'],
        }
        if cfg.use_wandb:
            videos = glob.glob(os.path.join(eval_tmp, 'videos', '*.mp4'))
            if videos:
                eval_log['eval/video'] = wandb.Video(videos[0], fps=30, format='mp4')
            wandb.log(eval_log, step=total_env_step)
            wandb.finish()
    finally:
        shutil.rmtree(eval_tmp, ignore_errors=True)
    print(f'\nDone. Total time: {(time.time() - start_time) / 60:.1f} min')


def load_cfg() -> OmegaConf:
    cfg = parse_cfg(CFG_PATH)
    cfg = OmegaConf.merge(OmegaConf.create(PHASED_DEFAULTS), cfg)

    custom_path = cfg.get('cfg', None)
    if custom_path:
        custom = OmegaConf.load(custom_path)
        cli    = OmegaConf.from_cli()
        cli_overrides = OmegaConf.create({k: v for k, v in cli.items() if k != 'cfg'})
        cfg = OmegaConf.merge(cfg, custom, cli_overrides)

    # Evaluate arithmetic strings (e.g. "1000/4" from OmegaConf interpolation)
    for k, v in cfg.items():
        if isinstance(v, str):
            m = re.match(r'^(\d+)([+\-*/])(\d+)$', v)
            if m:
                result = eval(m.group(1) + m.group(2) + m.group(3))
                cfg[k] = int(result) if isinstance(result, float) and result.is_integer() else result

    ar = cfg.action_repeat

    # Convert all MuJoCo step counts to agent steps
    cfg.train_steps         = int(cfg.mujoco_train_steps)         // ar
    cfg.seed_steps          = int(cfg.mujoco_seed_steps)          // ar
    cfg.decoder_start_steps = int(cfg.mujoco_decoder_start_steps) // ar
    cfg.latent_start_steps  = int(cfg.mujoco_latent_start_steps)  // ar

    # Rebuild schedule strings from MuJoCo step counts
    std_steps     = int(cfg.mujoco_std_schedule_steps)     // ar
    horizon_steps = int(cfg.mujoco_horizon_schedule_steps) // ar
    cfg.std_schedule     = f"linear(0.5, {cfg.min_std}, {std_steps})"
    cfg.horizon_schedule = f"linear(1, {cfg.horizon}, {horizon_steps})"

    assert cfg.seed_steps          < cfg.decoder_start_steps, \
        'mujoco_seed_steps must be < mujoco_decoder_start_steps'
    assert cfg.decoder_start_steps < cfg.latent_start_steps,  \
        'mujoco_decoder_start_steps must be < mujoco_latent_start_steps'
    assert cfg.latent_start_steps  <= cfg.train_steps,        \
        'mujoco_latent_start_steps must be <= mujoco_train_steps'

    return cfg


if __name__ == '__main__':
    cfg = load_cfg()
    train(cfg)
