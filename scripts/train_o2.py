"""
train_o2.py — Single-phase O2 training: TOLD + decoder trained jointly from scratch.

No phased mechanism: after the random seed-exploration steps, every episode is
collected with latent CEM planning (CEM_in_latent, through the decoder) and both
TOLD and the decoder are updated every episode.

Usage:
    python scripts/train_o2.py task=walker-walk seed=1
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
O2_DEFAULTS_PATH = REPO_ROOT / 'cfgs' / 'o2_default.yaml'

# Script-specific bookkeeping only — O2 algorithm/architecture defaults live
# in cfgs/o2_default.yaml (single source of truth, shared with
# train_o2_phased.py and with bare TDMPC_O2(cfg) construction, e.g. from a
# notebook).
DEFAULTS = {
    # Agent steps (post-action_repeat) — same units and convention as
    # tdmpc/cfgs/default.yaml's own train_steps/seed_steps. Override per-task
    # if needed, same as any other cfg value (e.g. seed_steps=2000, or
    # train_steps=40000/${action_repeat} to specify raw MuJoCo interactions).
    'train_steps': 10000,
    'seed_steps':  1000,   # random exploration before any updates/latent planning

    # Eval (periodic, no video; final eval below adds video)
    'eval_freq':     20000,  # env steps
    'eval_episodes': 5,

    # W&B
    'use_wandb':     True,
    'wandb_project': 'TDMPC_O2',
    'wandb_entity':  'odysseaskon-national-technical-university-of-athens',
    'exp_name':      'o2_joint',
}


def _upload_model(agent, label: str, metadata: dict) -> None:
    """Save model to a temp file, upload as W&B artifact, delete temp file."""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name
    try:
        agent.save(tmp_path)
        art = wandb.Artifact(name=label, type='model', metadata=metadata)
        art.add_file(tmp_path)
        wandb.log_artifact(art)
    finally:
        os.unlink(tmp_path)


def train(cfg):
    assert torch.cuda.is_available(), 'CUDA is required.'
    set_seed(cfg.seed)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.exp_name,
            name=f"{cfg.task}-seed{cfg.seed}",
            tags=[cfg.task, f"seed:{cfg.seed}"],
            notes=cfg.get('wandb_notes', None),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    env    = make_env(cfg)
    cfg.latent_action_dim = cfg.horizon * cfg.action_dim
    agent  = TDMPC_O2(cfg)
    buffer = ReplayBuffer(cfg)

    print('=' * 60)
    print(f'Task:                {cfg.task}')
    print(f'Train steps:         {cfg.train_steps:,}  ({int(cfg.train_steps * cfg.action_repeat):,} env steps)')
    print(f'  Seed ends at:      {cfg.seed_steps:,}  ({int(cfg.seed_steps * cfg.action_repeat):,} env steps)')
    print(f'TOLD updates/ep:     {cfg.told_updates}')
    print(f'Decoder updates/ep:  {cfg.decoder_updates}')
    print(f'Seed:                {cfg.seed}')
    print('=' * 60 + '\n')

    episode_idx = 0
    start_time  = time.time()
    task_safe   = cfg.task.replace('-', '_')

    def row(label, val):
        print(f'  {label:<22}: {val}')

    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
        # --- Collect one episode ---
        t_ep = time.time()
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            if step < cfg.seed_steps:
                action_np = env.action_space.sample()
                action = torch.tensor(action_np, dtype=torch.float32, device=agent.device)
            else:
                action, *_ = agent.CEM_in_latent(
                    obs, step=step, sample_final_action=True, t0=episode.first
                )
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
        buffer += episode
        episode_idx += 1
        ep_time = time.time() - t_ep

        # --- Updates ---
        train_metrics = {}
        dec_metrics   = {}
        update_time   = 0.0
        decoder_time  = 0.0
        if step >= cfg.seed_steps:
            # Same TOLD update cadence as tdmpc/src/train.py: a one-time catch-up
            # burst of seed_steps updates when first crossing the threshold, then
            # one update per environment decision (episode_length) every episode
            # after — instead of the fixed cfg.told_updates budget used elsewhere.
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length

            t_update = time.time()
            train_metrics = update_tdmpc(agent, buffer, step, num_updates=num_updates)
            update_time = time.time() - t_update

            t_dec = time.time()
            dec_metrics = update_decoder(agent, buffer, cfg, step)
            decoder_time = time.time() - t_dec

        env_step   = int(step * cfg.action_repeat)
        horizon    = int(linear_schedule(cfg.horizon_schedule, step))
        std        = linear_schedule(cfg.std_schedule, step)
        total_time = time.time() - start_time

        SEP = '─' * 42
        print(f'\n{SEP}')
        print(f'  Episode {episode_idx}   step {env_step:,}')
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

        log = {
            'episode':              episode_idx,
            'train/episode_reward': episode.cumulative_reward,
            'train/horizon':        horizon,
            'train/std':            std,
            **{f'train/{k}': v for k, v in train_metrics.items()},
            **{f'decoder/{k}': v for k, v in dec_metrics.items()},
            **{f'decoder/dcem_iter_{i}_grad': g for i, g in grad_tracker},
        }

        # --- Periodic evaluation ---
        if step >= cfg.seed_steps and env_step % cfg.eval_freq == 0:
            eval_metrics = evaluate_agent(
                env, agent, cfg, step=env_step, n_episodes=cfg.eval_episodes,
                policy='cem_latent', sample_final_action=True,
            )
            log['eval/mean_reward'] = eval_metrics['mean_reward']
            log['eval/std_reward']  = eval_metrics['std_reward']

        if cfg.use_wandb:
            wandb.log(log, step=env_step)

    # --- Final evaluation with video ---
    total_env_step = int(cfg.train_steps * cfg.action_repeat)
    eval_tmp = tempfile.mkdtemp()
    try:
        eval_metrics = evaluate_agent(
            env, agent, cfg, step=total_env_step, n_episodes=cfg.eval_episodes,
            save_dir=eval_tmp, video_mode='best_worst',
            use_latent=True, sample_final_action=True,
        )
        eval_log = {
            'eval/mean_reward': eval_metrics['mean_reward'],
            'eval/std_reward':  eval_metrics['std_reward'],
        }
        if cfg.use_wandb:
            best_videos  = glob.glob(os.path.join(eval_tmp, 'videos', '*_best_*.mp4'))
            worst_videos = glob.glob(os.path.join(eval_tmp, 'videos', '*_worst_*.mp4'))
            if best_videos:
                eval_log['eval/video_best'] = wandb.Video(best_videos[0], fps=30, format='mp4')
            if worst_videos:
                eval_log['eval/video_worst'] = wandb.Video(worst_videos[0], fps=30, format='mp4')
            wandb.log(eval_log, step=total_env_step)
            _upload_model(agent,
                label=f"final_{task_safe}_seed{cfg.seed}",
                metadata={'task': cfg.task, 'seed': cfg.seed,
                          'total_time_s': int(time.time() - start_time)})
            wandb.finish()
    finally:
        shutil.rmtree(eval_tmp, ignore_errors=True)
    print(f'\nDone. Total time: {(time.time() - start_time) / 60:.1f} min')


def make_cfg(task: str, **overrides) -> OmegaConf:
    """
    Build a config programmatically for use in notebooks.

    Example:
        from scripts.train_o2 import make_cfg
        cfg = make_cfg('walker-walk', seed=1, diversity_coeff=0.1, dec_grad_clip_norm=1)
        agent = TDMPC_O2(cfg)

    Any kwarg becomes a CLI-style override token, so it takes priority over
    both cfgs/o2_default.yaml and tdmpc/cfgs/default.yaml — same precedence
    a real `python scripts/train_o2.py task=... key=value` run gets. The
    returned cfg is a normal mutable OmegaConf object, so it can also be
    edited afterward (e.g. `cfg.horizon = 8`) before constructing the agent.
    """
    old_argv = sys.argv
    sys.argv = ['train_o2', f'task={task}'] + [f'{k}={v}' for k, v in overrides.items()]
    try:
        cfg = load_cfg()
    finally:
        sys.argv = old_argv
    return cfg


def load_cfg() -> OmegaConf:
    cfg = parse_cfg(CFG_PATH)
    cli = OmegaConf.from_cli()
    cli_overrides = OmegaConf.create({k: v for k, v in cli.items() if k != 'cfg'})

    # `cfg` (base default.yaml + task yaml + CLI, from parse_cfg) is lowest
    # priority here, so o2_default.yaml/DEFAULTS can override its non-CLI
    # (base/task) values — e.g. DEFAULTS' use_wandb=True over the base
    # yaml's use_wandb=false. But cfg also contains any real CLI overrides,
    # which DEFAULTS must not be allowed to clobber — so cli_overrides is
    # re-applied last, re-asserting whatever the user actually typed.
    cfg = OmegaConf.merge(cfg, OmegaConf.load(O2_DEFAULTS_PATH), OmegaConf.create(DEFAULTS), cli_overrides)

    custom_path = cfg.get('cfg', None)
    if custom_path:
        custom = OmegaConf.load(custom_path)
        cfg = OmegaConf.merge(cfg, custom, cli_overrides)

    # Evaluate arithmetic strings (e.g. "40000/4" from OmegaConf interpolation,
    # such as a custom yaml/CLI value written as train_steps=40000/${action_repeat})
    for k, v in cfg.items():
        if isinstance(v, str):
            m = re.match(r'^(\d+)([+\-*/])(\d+)$', v)
            if m:
                result = eval(m.group(1) + m.group(2) + m.group(3))
                cfg[k] = int(result) if isinstance(result, float) and result.is_integer() else result

    return cfg


if __name__ == '__main__':
    cfg = load_cfg()
    train(cfg)
