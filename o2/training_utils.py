"""
training_utils.py
-----------------
Shared training utilities used across all training scripts.

Functions:
    update_tdmpc     — update the TOLD world model (works with TDMPC and TDMPC_O2)
    update_decoder   — off-policy DDPG decoder update loop
"""

import random
import torch
import numpy as np
from algorithm.helper import linear_schedule


def sample_decoder_batch(buffer, batch_size, n=None, use_is_weights=False):
    """
    Sample one decoder-update batch at a specific batch size without permanently
    mutating the shared cfg (buffer.cfg IS agent.cfg — same object).

    Returns:
        obs:     [batch_size, obs_dim]
        weights: [batch_size] IS weights, or None if use_is_weights=False
    """
    old = buffer.cfg.batch_size
    buffer.cfg.batch_size = batch_size
    if use_is_weights:
        obs, _, _, _, _, weights = buffer.sample()
    else:
        obs     = sample_recent_obs(buffer, n) if n else buffer.sample()[0]
        weights = None
    buffer.cfg.batch_size = old
    return obs, weights


def sample_recent_obs(buffer, n):
    """
    Uniformly sample a batch of observations from the n most recent transitions.

    Args:
        buffer: ReplayBuffer instance
        n:      number of most recent transitions to sample from

    Returns:
        obs: [batch_size, obs_dim] observation tensor
    """
    total = int(buffer._full) * buffer.capacity + (not buffer._full) * buffer.idx
    n = min(n, total)
    end   = buffer.idx
    start = (end - n) % buffer.capacity

    rel_idxs = torch.randint(0, n, (buffer.cfg.batch_size,), device=buffer.device)
    idxs = (rel_idxs + start) % buffer.capacity
    return buffer._get_obs(buffer._obs, idxs)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_tdmpc(agent, buffer, step):
    """
    Update the TOLD world model for cfg.told_updates iterations.

    Works with both TDMPC and TDMPC_O2. When used with TDMPC_O2, ensures
    TOLD gradients are enabled and decoder gradients are disabled for the
    duration of the update.

    Args:
        agent:  TDMPC or TDMPC_O2 instance
        buffer: ReplayBuffer
        step:   current global step

    Returns:
        dict of mean loss metrics across all update iterations
    """
    if hasattr(agent.model, 'track_TOLD_grad'):
        agent.model.track_TOLD_grad(True)
        agent.model.track_O2_grad(False)

    buffer.cfg.batch_size = agent.cfg.batch_size
    num_updates = getattr(agent.cfg, 'told_updates', agent.cfg.episode_length)
    metrics = {}
    for i in range(num_updates):
        update_metrics = agent.update(buffer, step + i)
        for k, v in update_metrics.items():
            metrics[k] = metrics.get(k, 0.0) + v
    for k in metrics:
        metrics[k] /= agent.cfg.told_updates

    if hasattr(agent.model, 'track_O2_grad'):
        agent.model.track_O2_grad(True)

    return metrics


def update_decoder(agent, buffer, cfg, step):
    """
    Off-policy DDPG decoder update loop.

    Freezes TOLD, samples batches from the buffer, runs DCEMethod_v2 to get
    differentiable latent action means, then calls update_decoder_DDPG.

    Args:
        agent:  TDMPC_O2 instance
        buffer: ReplayBuffer
        cfg:    OmegaConf config
        step:   current global step

    Returns:
        dict with averaged metrics across all update iterations, plus
        grad_tracker (from last iteration) and decoder_grad_norm_max.
    """
    agent.model.track_TOLD_grad(False)
    horizon = int(linear_schedule(cfg.horizon_schedule, step))

    n              = getattr(agent.cfg, 'dcem_sampling_n', None)
    use_is_weights = getattr(agent.cfg, 'use_is_weights', False)
    accum             = {}
    grad_norm_max     = 0.0
    last_grad_tracker = []
    for _ in range(agent.cfg.decoder_updates):
        obs, weights = sample_decoder_batch(buffer, agent.cfg.dcem_batch_size,
                                            n=n, use_is_weights=use_is_weights)
        _, u_mean, _, _, _, grad_tracker, diversity, _ = agent.DCEM(obs, step=step)
        metrics = agent.update_decoder_DDPG(obs, u_mean, horizon, weights)
        metrics.update(diversity)
        grad_norm_max = max(grad_norm_max, metrics['decoder_grad_norm'])
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        last_grad_tracker = grad_tracker

    agent.model.track_TOLD_grad(True)
    n_updates = agent.cfg.decoder_updates
    return {k: v / n_updates for k, v in accum.items()} | {'grad_tracker': last_grad_tracker, 'decoder_grad_norm_max': grad_norm_max}
