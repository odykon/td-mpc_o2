"""
training_utils.py
-----------------
Shared training utilities used across all training scripts.

Functions:
    update_tdmpc       — update the TOLD world model (works with TDMPC and TDMPC_O2)
    update_decoder     — off-policy DDPG decoder update loop
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


def update_tdmpc(agent, buffer, step, num_updates=None):
    """
    Update the TOLD world model for num_updates iterations (default: cfg.told_updates).

    Works with both TDMPC and TDMPC_O2. When used with TDMPC_O2, ensures
    TOLD gradients are enabled and decoder gradients are disabled for the
    duration of the update.

    Args:
        agent:       TDMPC or TDMPC_O2 instance
        buffer:      ReplayBuffer
        step:        current global step
        num_updates: number of gradient updates to run. Defaults to
                     agent.cfg.told_updates (fixed per-episode budget). Pass
                     an explicit value to match tdmpc/src/train.py's own
                     convention instead (cfg.seed_steps on the first update,
                     cfg.episode_length on every one after).

    Returns:
        dict of mean loss metrics across all update iterations
    """
    if hasattr(agent.model, 'track_TOLD_grad'):
        agent.model.track_TOLD_grad(True)
        agent.model.track_O2_grad(False)

    buffer.cfg.batch_size = agent.cfg.batch_size
    if num_updates is None:
        num_updates = agent.cfg.told_updates
    metrics = {}
    for i in range(num_updates):
        update_metrics = agent.update(buffer, step + i)
        for k, v in update_metrics.items():
            metrics[k] = metrics.get(k, 0.0) + v
    for k in metrics:
        metrics[k] /= num_updates

    if hasattr(agent.model, 'track_O2_grad'):
        agent.model.track_O2_grad(True)

    return metrics


def update_decoder(agent, buffer, cfg, step):
    """
    Off-policy decoder update loop.

    Freezes TOLD, samples batches from the buffer, runs DCEM to get
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

    use_is_weights    = agent.cfg.use_is_weights
    uniform_sampling  = agent.cfg.decoder_uniform_sampling
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
