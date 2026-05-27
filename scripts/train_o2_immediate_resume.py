"""
train_o2_immediate_resume.py — O2 immediate training resumed from a phased checkpoint.

Downloads the intermediate model + buffer artifacts saved by train_o2_phased.py
at mujoco_latent_start_steps, then continues with latent CEM + TOLD + decoder DDPG
updates from that point. The decoder is initialised fresh — only the TOLD weights
(encoder, dynamics, reward, Q, pi) are loaded from the checkpoint.

W&B env_step starts at mujoco_step_offset so curves overlay directly with the
corresponding phased run in the same group.

Usage:
    python scripts/train_o2_immediate_resume.py task=walker-walk seed=1
    python scripts/train_o2_immediate_resume.py cfg=cfgs/exp_immediate_resume.yaml task=walker-walk seed=1
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

RESUME_DEFAULTS = {
    # Checkpoint origin — must match mujoco_latent_start_steps in the phased run
    'mujoco_step_offset':  20000,
    'mujoco_resume_steps': 20000,

    # Schedule endpoints in MuJoCo steps — set to match the phased run so
    # std / horizon schedules are continuous across the boundary
    'mujoco_std_schedule_steps':     10000,
    'mujoco_horizon_schedule_steps': 10000,

    # Update cadence
    'told_updates':    500,
    'decoder_updates': 100,

    # O2 architecture
    'latent_action_dim':  128,
    'decoder_init':       True,   # fresh decoder initialisation
    'use_latent_state':   True,
    'dcem_batch_size':    64,
    'latent_num_samples': 32,
    'latent_num_elites':  8,
    'lml_temperature':    1,
    'dcem_sampling_n':    None,
    'saturation_coeff':   0.0,
    'use_is_weights':     True,
    'dec_grad_clip_norm': 20,

    # Eval (one at end with video)
    'eval_episodes': 5,

    # W&B
    'wandb_project': 'TDMPC_O2',
    'wandb_entity':  'odysseaskon-national-technical-university-of-athens',
    'exp_name':      'o2_immediate_resume',

    # Required by TDMPC_O2 init; set to 0 so o2 mode is active from step_offset
    'decoder_start_steps': 0,
    'latent_start_steps':  0,
    # No seed phase — model is already trained
    'seed_steps': 0,
}

# Keys belonging to the decoder and value network — excluded when loading TOLD weights
_DECODER_KEY_PREFIXES = ('_action_decoder.', '_V.')


def _strip_decoder_keys(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items()
            if not any(k.startswith(p) for p in _DECODER_KEY_PREFIXES)}


def _upload_model(agent, label: str, metadata: dict) -> None:
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name
    try:
        agent.save(tmp_path)
        art = wandb.Artifact(name=label, type='model', metadata=metadata)
        art.add_file(tmp_path)
        wandb.log_artifact(art)
    finally:
        os.unlink(tmp_path)


def _upload_buffer(buffer, label: str, metadata: dict) -> None:
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        tmp_path = f.name
    try:
        torch.save(buffer.__dict__, tmp_path)
        art = wandb.Artifact(name=label, type='buffer', metadata=metadata)
        art.add_file(tmp_path)
        wandb.log_artifact(art)
    finally:
        os.unlink(tmp_path)


def train(cfg):
    assert torch.cuda.is_available(), 'CUDA is required.'
    set_seed(cfg.seed)

    task_safe = cfg.task.replace('-', '_')

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

    # ── Download intermediate model (TOLD only) + buffer from W&B ────────────
    api = wandb.Api()

    model_tmp = tempfile.mkdtemp()
    try:
        art = api.artifact(
            f"{cfg.wandb_entity}/{cfg.wandb_project}/"
            f"intermediate_{task_safe}_seed{cfg.seed}:latest"
        )
        art.download(root=model_tmp)
        model_file = glob.glob(os.path.join(model_tmp, '*.pt'))[0]
        ckpt = torch.load(model_file, map_location=agent.device)

        model_sd  = ckpt['model']        if 'model'        in ckpt else ckpt
        target_sd = ckpt['model_target'] if 'model_target' in ckpt else None

        agent.model.load_state_dict(_strip_decoder_keys(model_sd), strict=False)
        if target_sd is not None:
            agent.model_target.load_state_dict(_strip_decoder_keys(target_sd), strict=False)
        else:
            agent.model_target.load_state_dict(agent.model.state_dict())
        print(f'Loaded TOLD weights from intermediate checkpoint (decoder kept fresh).')
    finally:
        shutil.rmtree(model_tmp, ignore_errors=True)

    buffer_tmp = tempfile.mkdtemp()
    try:
        art = api.artifact(
            f"{cfg.wandb_entity}/{cfg.wandb_project}/"
            f"intermediate_buffer_{task_safe}_seed{cfg.seed}:latest"
        )
        art.download(root=buffer_tmp)
        buffer_file = glob.glob(os.path.join(buffer_tmp, '*.pth'))[0]
        buffer.__dict__.update(torch.load(buffer_file, weights_only=False))
        print(f'Loaded intermediate buffer (seed {cfg.seed}).')
    finally:
        shutil.rmtree(buffer_tmp, ignore_errors=True)

    print('=' * 60)
    print(f'Task:               {cfg.task}')
    print(f'Resuming from:      {cfg.mujoco_step_offset:,}  MuJoCo')
    print(f'Running for:        {cfg.mujoco_resume_steps:,}  MuJoCo')
    print(f'TOLD updates/ep:    {cfg.told_updates}')
    print(f'Decoder updates/ep: {cfg.decoder_updates}')
    print(f'Seed:               {cfg.seed}')
    print('=' * 60 + '\n')

    episode_idx = 0
    start_time  = time.time()

    for step in range(cfg.step_offset,
                      cfg.step_offset + cfg.resume_steps + cfg.episode_length,
                      cfg.episode_length):

        # Collect episode with latent CEM
        t_ep = time.time()
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            action, *_ = agent.CEM_in_latent(
                obs, step=step, t0=episode.first, sample_final_action=True
            )
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
        buffer += episode
        episode_idx += 1
        ep_time = time.time() - t_ep

        # Updates
        t_update = time.time()
        train_metrics = update_tdmpc(agent, buffer, step)
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
        print(f'  Episode {episode_idx}   step {env_step:,}   [o2]')
        print(SEP)
        def row(label, val):
            print(f'  {label:<22}: {val}')
        row('Reward',       f'{episode.cumulative_reward:>10.1f}')
        row('Horizon',      f'{horizon:>10d}')
        row('Std',          f'{std:>10.3f}')
        row('Ep time',      f'{ep_time:>9.1f}s')
        row('Update time',  f'{update_time:>9.1f}s')
        row('Decoder time', f'{decoder_time:>9.1f}s')
        grad_tracker = dec_metrics.pop('grad_tracker', [])
        for k, v in dec_metrics.items():
            row(k, f'{v:>10.4f}')
        row('Total time',   f'{total_time:>9.0f}s')
        for k, v in train_metrics.items():
            row(k, f'{v:>10.4f}')

        wandb.log({
            'episode':              episode_idx,
            'train/episode_reward': episode.cumulative_reward,
            'train/horizon':        horizon,
            'train/std':            std,
            **{f'train/{k}': v for k, v in train_metrics.items()},
            **{f'decoder/{k}': v for k, v in dec_metrics.items()},
            **{f'decoder/dcem_iter_{i}_grad': g for i, g in grad_tracker},
        }, step=env_step)

    # ── Final evaluation with video ──────────────────────────────────────────
    total_env_step = int((cfg.step_offset + cfg.resume_steps) * cfg.action_repeat)
    eval_tmp = tempfile.mkdtemp()
    try:
        eval_metrics = evaluate_agent(
            env, agent, cfg,
            step=total_env_step,
            n_episodes=cfg.eval_episodes,
            save_dir=eval_tmp,
            video_mode='first',
        )
        eval_log = {
            'eval/mean_reward': eval_metrics['mean_reward'],
            'eval/std_reward':  eval_metrics['std_reward'],
        }
        videos = glob.glob(os.path.join(eval_tmp, 'videos', '*.mp4'))
        if videos:
            eval_log['eval/video'] = wandb.Video(videos[0], fps=30, format='mp4')
        wandb.log(eval_log, step=total_env_step)
    finally:
        shutil.rmtree(eval_tmp, ignore_errors=True)

    # ── Upload final model + buffer ───────────────────────────────────────────
    _upload_model(agent,
        label=f"final_{task_safe}_seed{cfg.seed}",
        metadata={'task': cfg.task, 'seed': cfg.seed,
                  'total_time_s': int(time.time() - start_time)})

    _upload_buffer(buffer,
        label=f"buffer_{task_safe}_seed{cfg.seed}",
        metadata={'task': cfg.task, 'seed': cfg.seed})

    wandb.finish()
    print(f'\nDone. Total time: {(time.time() - start_time) / 60:.1f} min')


def load_cfg() -> OmegaConf:
    cfg = parse_cfg(CFG_PATH)
    cfg = OmegaConf.merge(OmegaConf.create(RESUME_DEFAULTS), cfg)

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

    cfg.step_offset  = int(cfg.mujoco_step_offset)  // ar
    cfg.resume_steps = int(cfg.mujoco_resume_steps) // ar
    cfg.train_steps  = cfg.step_offset + cfg.resume_steps  # for buffer capacity + schedules

    std_steps     = int(cfg.mujoco_std_schedule_steps)     // ar
    horizon_steps = int(cfg.mujoco_horizon_schedule_steps) // ar
    cfg.std_schedule     = f"linear(0.5, {cfg.min_std}, {std_steps})"
    cfg.horizon_schedule = f"linear(1, {cfg.horizon}, {horizon_steps})"

    assert cfg.mujoco_step_offset > 0, 'mujoco_step_offset must be > 0'
    assert cfg.mujoco_resume_steps > 0, 'mujoco_resume_steps must be > 0'

    return cfg


if __name__ == '__main__':
    cfg = load_cfg()
    train(cfg)
