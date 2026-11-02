"""
dcem.py
-------
Base differentiable CEM in latent action space (Amos & Yarats, 2020):
sample u ~ N(mean, std), decode, score with the TOLD model, weight the
candidates, refit mean/std from the weights. No value normalization, no
momentum, no std clamping.

Candidate weighting, cfg.dcem_weighting (weights over the N samples sum to 1):
    lml          (default) soft top-k:  LML(N=K)(lml_temperature * v) / K
    lml_softmax  soft top-k, then TD-MPC-style reweighting within it:
                 w ∝ p_LML * exp(dcem_softmax_temperature * v)
    softmax      MPPI-style over all candidates: softmax(dcem_softmax_temperature * v)
"""

import torch

import algorithm.helper as h
from lml import LML


def estimate_value_with_grad(agent, z, actions, horizon, return_parts=False):
    """TDMPC.estimate_value without @torch.no_grad(). return_parts: also return (reward sum, terminal Q)."""
    G, discount = 0, 1
    for t in range(horizon):
        z, reward = agent.model.next(z, actions[t])
        G = G + discount * reward
        discount *= agent.cfg.discount
    q_term = discount * torch.min(*agent.model.Q(z, agent.model.pi(z, agent.cfg.min_std)))
    if return_parts:
        return G + q_term, G, q_term
    return G + q_term


def dcem(agent, obs, step, init_std=1.0):
    """
    Args:
        obs:  [B, obs_dim] observation batch.
        step: global step (for the horizon schedule).

    Returns:
        u_mean, u_std: [B, latent_action_dim] final search distribution,
                       differentiable w.r.t. the decoder parameters.
        horizon:       the planning horizon used.

    Also sets agent.dcem_stats (first/last iteration): value_std is the spread
    of candidate values LML sees, eff_elites is 1 / sum(w^2), which goes from
    K (hard top-k) to N (uniform weights).
    """
    cfg = agent.cfg
    B, N, K, d = obs.shape[0], cfg.latent_num_samples, cfg.latent_num_elites, cfg.latent_action_dim
    horizon = int(min(cfg.horizon, h.linear_schedule(cfg.horizon_schedule, step)))
    weighting = cfg.get('dcem_weighting', 'lml')
    tau = cfg.get('dcem_softmax_temperature', 0.5)

    z = agent.model.h(obs).detach()
    z_rep = z.repeat_interleave(N, dim=0)                                   # [B*N, z_dim]

    u_mean = torch.zeros(B, d, device=agent.device)
    u_std = init_std * torch.ones(B, d, device=agent.device)
    value_stds, eff_elites = [], []

    for _ in range(cfg.dcem_iterations):
        u = u_mean.unsqueeze(1) + u_std.unsqueeze(1) * torch.randn(B, N, d, device=agent.device)
        actions = agent.decoder(u.view(B * N, d), z_rep)                    # [H, B*N, A]
        value = estimate_value_with_grad(agent, z_rep, actions, horizon).view(B, N)

        if weighting == 'lml':
            w = LML(N=K, verbose=0)(value * cfg.lml_temperature) / K        # [B, N], sums to 1
        elif weighting == 'lml_softmax':
            p = LML(N=K, verbose=0)(value * cfg.lml_temperature)            # [B, N] in [0, 1], sums to K
            w = torch.softmax(p.clamp_min(1e-12).log() + tau * value, dim=1)
        elif weighting == 'softmax':
            w = torch.softmax(tau * value, dim=1)
        else:
            raise ValueError(f"Unknown dcem_weighting: {weighting!r} (expected 'lml', 'lml_softmax' or 'softmax')")
        value_stds.append(value.detach().std(dim=1).mean().item())
        eff_elites.append((1 / w.detach().pow(2).sum(dim=1)).mean().item())
        w = w.unsqueeze(-1)
        u_mean = (w * u).sum(dim=1)
        u_std = ((w * (u - u_mean.unsqueeze(1)) ** 2).sum(dim=1) + 1e-6).sqrt()

    agent.dcem_stats = {
        'value_std_first': value_stds[0], 'value_std_last': value_stds[-1],
        'eff_elites_first': eff_elites[0], 'eff_elites_last': eff_elites[-1],
        'u_std_final': u_std.mean().item(),
    }
    return u_mean, u_std, horizon
