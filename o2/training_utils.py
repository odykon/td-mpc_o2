"""
training_utils.py
-----------------
Shared training utilities used across all training scripts.

Functions:
    update_tdmpc       — update the TOLD world model (works with TDMPC and TDMPC_O2)
    update_decoder     — off-policy decoder update loop (stochastic/SAC-style)
    finalize_decoder   — standalone deterministic decoder finalization pass
"""

import random
import torch
import numpy as np
from algorithm.helper import linear_schedule


def sample_decoder_batch(buffer, batch_size, use_is_weights=False, uniform=False):
    """
    Sample one decoder-update batch at a specific batch size without permanently
    mutating the shared cfg (buffer.cfg IS agent.cfg — same object).

    If uniform=True, samples uniformly over the buffer instead of through
    TOLD's prioritized replay: the decoder update only reads `obs` and rolls
    forward through the learned dynamics model (estimate_value_GAE), so it has
    no dependency on PER's TD-error-driven priorities. use_is_weights is
    ignored in this case (uniform sampling has no bias to correct). TOLD's own
    updates (update_tdmpc) are unaffected either way — they call
    buffer.sample() directly.

    Toggle via cfg.decoder_uniform_sampling (see update_decoder below).

    Returns:
        obs:     [batch_size, obs_dim]
        weights: [batch_size] IS weights, or None if uniform or use_is_weights=False
    """
    old = buffer.cfg.batch_size
    buffer.cfg.batch_size = batch_size
    if uniform:
        total   = buffer.idx if not buffer._full else buffer.capacity
        idxs    = torch.randint(0, total, (batch_size,), device=buffer.device)
        obs     = buffer._get_obs(buffer._obs, idxs)
        weights = None
    elif use_is_weights:
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
    differentiable latent action means, then calls update_decoder_stoch.

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
    uniform_sampling  = getattr(agent.cfg, 'decoder_uniform_sampling', False)
    accum             = {}
    grad_norm_max     = 0.0
    last_grad_tracker = []
    for _ in range(agent.cfg.decoder_updates):
        obs, weights = sample_decoder_batch(buffer, agent.cfg.dcem_batch_size,
                                            use_is_weights=use_is_weights, uniform=uniform_sampling)
        _, u_mean, u_std, _, _, grad_tracker, diversity = agent.DCEM(obs, step=step)
        metrics = agent.update_decoder_stoch(obs, u_mean, horizon, weights, u_std=u_std)
        metrics.update(diversity)
        grad_norm_max = max(grad_norm_max, metrics['decoder_grad_norm'])
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        last_grad_tracker = grad_tracker

    agent.model.track_TOLD_grad(True)
    n_updates = agent.cfg.decoder_updates
    return {k: v / n_updates for k, v in accum.items()} | {'grad_tracker': last_grad_tracker, 'decoder_grad_norm_max': grad_norm_max}


def finalize_decoder(agent, buffer, cfg, step, num_updates):
    """
    Deterministic decoder finalization pass.

    Same DCEM → decoder-update loop as update_decoder, but calls
    update_decoder_det instead of update_decoder_stoch: no reparameterized
    sampling, no GMM diversity fit, no saturation penalty — just u_mean
    optimized directly against the GAE value estimate. Intended to be called
    once, standalone (e.g. from a notebook), to polish the decoder at the end
    of training rather than as part of the per-episode training loop.

    Args:
        agent:       TDMPC_O2 instance
        buffer:      ReplayBuffer
        cfg:         OmegaConf config
        step:        step to evaluate horizon_schedule at
        num_updates: number of DCEM + decoder-update iterations to run

    Returns:
        dict with averaged metrics across all update iterations.
    """
    agent.model.track_TOLD_grad(False)
    horizon = int(linear_schedule(cfg.horizon_schedule, step))
    use_is_weights   = getattr(agent.cfg, 'use_is_weights', False)
    uniform_sampling = getattr(agent.cfg, 'decoder_uniform_sampling', False)
    accum = {}
    for _ in range(num_updates):
        obs, weights = sample_decoder_batch(buffer, agent.cfg.dcem_batch_size,
                                            use_is_weights=use_is_weights, uniform=uniform_sampling)
        _, u_mean, _, _, _, _, diversity = agent.DCEM(obs, step=step)
        metrics = agent.update_decoder_det(obs, u_mean, horizon, weights)
        metrics.update(diversity)
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v

    agent.model.track_TOLD_grad(True)
    return {k: v / num_updates for k, v in accum.items()}
