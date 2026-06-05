"""
training_utils.py
-----------------
Shared training utilities used across all training scripts.

Functions:
    update_tdmpc       — update the TOLD world model (works with TDMPC and TDMPC_O2)
    update_decoder     — off-policy DDPG decoder update loop
    update_decoder_pg  — on-policy PG decoder update loop
"""

import random
import torch
import numpy as np
from algorithm.helper import linear_schedule


def sample_decoder_batch(buffer, batch_size, use_is_weights=False):
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
        obs     = buffer.sample()[0]
        weights = None
    buffer.cfg.batch_size = old
    return obs, weights


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

    use_is_weights    = getattr(agent.cfg, 'use_is_weights', False)
    accum             = {}
    grad_norm_max     = 0.0
    last_grad_tracker = []
    for _ in range(agent.cfg.decoder_updates):
        obs, weights = sample_decoder_batch(buffer, agent.cfg.dcem_batch_size,
                                            use_is_weights=use_is_weights)
        _, u_mean, _, _, _, grad_tracker, diversity, log_det_loss = agent.DCEM(obs, step=step)
        metrics = agent.update_decoder_DDPG(obs, u_mean, horizon, weights, log_det_loss=log_det_loss)
        metrics.update(diversity)
        grad_norm_max = max(grad_norm_max, metrics['decoder_grad_norm'])
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        last_grad_tracker = grad_tracker

    agent.model.track_TOLD_grad(True)
    n_updates = agent.cfg.decoder_updates
    return {k: v / n_updates for k, v in accum.items()} | {'grad_tracker': last_grad_tracker, 'decoder_grad_norm_max': grad_norm_max}


def update_decoder_pg(agent, episode, step, alpha_v=0.0):
    """
    On-policy PG decoder update loop.

    Iterates over batches from the latest PGEpisode (on-policy transitions),
    runs DCEM on each batch of observations to get the current latent
    distribution, then calls PG_withV with the diversity loss from DCEM
    as the entropy bonus.

    Args:
        agent:   TDMPC_O2 instance
        episode: PGEpisode (finalized) from the most recent rollout
        step:    current global step
        alpha_v: entropy coefficient (e.g. from a variance schedule)

    Returns:
        dict with averaged metrics across all update iterations.
    """
    agent.model.track_TOLD_grad(False)

    accum             = {}
    n_batches         = 0
    last_grad_tracker = []
    for obs, reward, obs_t1, latent_action in episode.sample_batches(
            batch_size=agent.cfg.dcem_batch_size, shuffle=True):

        with torch.no_grad():
            z_t  = agent.model.h(obs)
            z_t1 = agent.model.h(obs_t1)
        _, u_mean, u_std, _, _, grad_tracker, diversity, log_det_loss = agent.DCEM(obs, step=step)

        pg_metrics = agent.PG_withV(z_t, z_t1, u_mean, u_std, reward, latent_action,
                                    alpha_v, log_det_loss=log_det_loss)
        v_loss     = agent.V_net_update(reward, z_t, z_t1)

        metrics = {**pg_metrics, 'v_loss': v_loss.item(), **diversity}
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + (v.item() if hasattr(v, 'item') else v)
        last_grad_tracker = grad_tracker
        n_batches += 1

    agent.model.track_TOLD_grad(True)
    n = max(n_batches, 1)
    return {k: v / n for k, v in accum.items()} | {'grad_tracker': last_grad_tracker}
