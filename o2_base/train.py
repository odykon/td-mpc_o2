"""
train.py
--------
One o2_base run (same as o2_base_training.ipynb):
    phase 1: standard TD-MPC (random actions during seed steps)
    switch:  pretrain the decoder for PRETRAIN_UPDATES updates
    phase 2: act with the decoder (DCEM), keep training TOLD + decoder

Usage:
    python o2_base/train.py --action_repeat 2 --seed 1
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tdmpc' / 'src'))
os.environ.setdefault('MUJOCO_GL', 'egl')

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from env import make_env
from algorithm.helper import Episode, ReplayBuffer
from scripts.train_o2 import make_cfg
from o2.training_utils import set_seed
from o2_base.agent import TDMPC_Dec

# Counted in episodes (each episode is 1000 // action_repeat agent steps)
PHASE1_EPISODES  = 20
PHASE2_EPISODES  = 20
SEED_EPISODES    = 4
PRETRAIN_UPDATES = 100
LATENT_ACTION_DIM = 100

DEC_FIELDS = ['ep', 'step', 'buffer_reward', 'decoder_loss', 'value_mean', 'reward_part', 'q_part',
              'grad_total', 'grad_direct', 'grad_unroll', 'grad_u_mean',
              'value_std_first', 'value_std_last', 'eff_elites_first', 'eff_elites_last', 'u_std_final']


@torch.no_grad()
def decoder_act(agent, obs, step):
    """Run DCEM (no grad), sample u from the final distribution, decode, return the first action."""
    obs = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    u_mean, u_std, _ = agent.dcem(obs, step)
    u = u_mean + u_std * torch.randn_like(u_mean)
    return agent.decoder(u, agent.model.h(obs))[0, 0]


def decoder_update_batch(agent, buffer, ep, step, dec_writer):
    obs, _, _, reward, *_ = buffer.sample()
    metrics = agent.update_decoder(obs[:agent.cfg.dcem_batch_size], step)
    dec_writer.writerow({'ep': ep, 'step': step, 'buffer_reward': reward.mean().item(), **metrics})
    return metrics


def save(agent, path):
    torch.save({**agent.state_dict(), 'decoder': agent.decoder.state_dict()}, path)


def train(task, action_repeat, seed, tdmpc_only=False, tag=''):
    cfg = make_cfg(
        task, seed=seed, action_repeat=action_repeat,
        horizon=5,
        dec_hidden_dim=256,
        dec_num_layers=2,
        latent_num_samples=32,
        latent_num_elites=8,
        iterations=5,
        dcem_iterations=5,
        lml_temperature=5,
        dcem_batch_size=64,
        decoder_updates=70,
    )
    cfg.episode_length = 1000 // cfg.action_repeat   # 1000/3 would otherwise be a float
    phase1_steps = PHASE1_EPISODES * cfg.episode_length
    cfg.seed_steps  = SEED_EPISODES * cfg.episode_length
    cfg.train_steps = (PHASE1_EPISODES + PHASE2_EPISODES) * cfg.episode_length
    cfg.horizon_schedule = f'linear(1,5,{phase1_steps})'
    cfg.std_schedule     = f'linear(0.5,0.05,{phase1_steps})'

    run_name = f'{task}_ar{action_repeat}_s{seed}{"_tdmpc" if tdmpc_only else ""}{f"_{tag}" if tag else ""}'
    out_dir = REPO_ROOT / 'runs' / 'o2_base' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    env = make_env(cfg)
    cfg.latent_action_dim = LATENT_ACTION_DIM
    agent  = TDMPC_Dec(cfg)
    buffer = ReplayBuffer(cfg)

    csv_file = open(out_dir / 'metrics.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'env_step', 'phase', 'reward', 'value_loss', 'decoder_loss', 'time'])

    dec_csv_file = open(out_dir / 'decoder_metrics.csv', 'w', newline='')
    dec_writer = csv.DictWriter(dec_csv_file, fieldnames=DEC_FIELDS)
    dec_writer.writeheader()

    rewards = []
    start_time = time.time()

    for ep in range(PHASE1_EPISODES + PHASE2_EPISODES):
        step = ep * cfg.episode_length
        phase2 = ep >= PHASE1_EPISODES and not tdmpc_only

        # --- Switch: save phase-1 model, pretrain the decoder ---
        if ep == PHASE1_EPISODES and not tdmpc_only:
            save(agent, out_dir / 'model_switch.pt')
            for _ in range(PRETRAIN_UPDATES):
                dec_metrics = decoder_update_batch(agent, buffer, ep, step, dec_writer)
            print(f'decoder pretrained: decoder_loss {dec_metrics["decoder_loss"]:.3f}')

        # --- Collect one episode ---
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            if phase2:
                action = decoder_act(agent, obs, step)
            else:
                action = agent.plan(obs, step=step, t0=episode.first)   # random during seed steps
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
        buffer += episode

        # --- TOLD updates (both phases) ---
        told_metrics = {}
        if ep >= SEED_EPISODES:
            num_updates = cfg.seed_steps if ep == SEED_EPISODES else cfg.episode_length
            for i in range(num_updates):
                told_metrics = agent.update(buffer, step + i)

        # --- Decoder updates (phase 2) ---
        dec_metrics = {}
        if phase2:
            for _ in range(cfg.decoder_updates):
                dec_metrics = decoder_update_batch(agent, buffer, ep, step, dec_writer)

        row = [step, step * cfg.action_repeat, 2 if phase2 else 1, episode.cumulative_reward,
               told_metrics.get('value_loss', ''), dec_metrics.get('decoder_loss', ''),
               round(time.time() - start_time)]
        writer.writerow(row)
        csv_file.flush()
        dec_csv_file.flush()
        rewards.append(episode.cumulative_reward)
        print(f'step {step:>6}  phase {row[2]}  reward {row[3]:7.1f}  time {row[6]:>6}s')

    save(agent, out_dir / 'model_final.pt')
    csv_file.close()
    dec_csv_file.close()

    plt.plot(rewards)
    plt.axvline(PHASE1_EPISODES, color='k', linestyle='--')
    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.title(f'{task} ar{action_repeat} s{seed}')
    plt.savefig(out_dir / 'reward.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='cheetah-run')
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--tdmpc_only', action='store_true', help='never switch to phase 2 (plain TD-MPC baseline)')
    parser.add_argument('--tag', default='', help='suffix for the run folder, so earlier results are not overwritten')
    args = parser.parse_args()
    train(args.task, args.action_repeat, args.seed, args.tdmpc_only, args.tag)
