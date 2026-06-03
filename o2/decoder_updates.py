"""
decoder_updates.py
------------------
Off-policy DDPG-style decoder update with GAE-weighted multi-horizon value targets.
"""

import torch
import torch.nn.utils as utils


def update_decoder_DDPG(self, obs, u_mean, horizon, weights=None, lambda_gae=None, log_det_loss=None):
    """
    DDPG-style decoder update using a GAE-weighted sum of n-step TD targets.

    V_λ = Σ_h w_h * V_h, where
        w_h = (1 - λ) * λ^(h-1)  for h < gae_horizons
        w_H = λ^(H-1)             (absorbs remaining weight; all weights sum to 1)

    Config: lambda_gae (default 0.5), gae_horizons (default 5).
    Saturation is monitored but not penalised.

    Args:
        obs:     [B, obs_dim] observation batch from the replay buffer.
        u_mean:  [B, latent_action_dim] differentiable latent action mean
                 obtained from DCEM().
        horizon: int planning horizon (used for gradient scaling).
        weights: optional [B] importance-sampling weights.

    Returns:
        dict with keys: decoder_loss, decoder_grad_norm, value_mean,
                        saturation, z_norm, u_norm, hidden_norm
    """
    self.action_dec_optim.zero_grad()

    z                 = self.model.h(obs).detach()
    z_norm            = z.norm(dim=-1).mean().item()
    u_norm            = u_mean.norm(dim=-1).mean().item()
    sequence, pretanh = self.model.decode_sequence(u_mean, z, return_pretanh=True)
    saturation        = pretanh.abs().mean().item()

    lam          = lambda_gae if lambda_gae is not None else 0.6
    gae_horizons = getattr(self.cfg, 'gae_horizons', 5)

    gae_weights = [(1 - lam) * lam**(h - 1) for h in range(1, gae_horizons)]
    gae_weights.append(lam**(gae_horizons - 1))

    value = sum(
        w * self.estimate_value_with_grad(z, sequence, h).nan_to_num(0).squeeze(-1)
        for w, h in zip(gae_weights, range(1, gae_horizons + 1))
    )

    per_sample_cost = -value
    if weights is not None:
        cost = (per_sample_cost * weights).mean()
    else:
        cost = per_sample_cost.mean()

    diversity_coeff = getattr(self.cfg, 'diversity_coeff', 0.0)
    if log_det_loss is not None and diversity_coeff > 0:
        cost = cost + diversity_coeff * log_det_loss

    cost.register_hook(lambda grad: grad * (1 / horizon))
    cost.backward()
    grad_norm = torch.sqrt(sum(
        p.grad.norm() ** 2
        for p in self.model._action_decoder.parameters() if p.grad is not None
    ))
    dec_grad_clip = getattr(self.cfg, 'dec_grad_clip_norm', None)
    if dec_grad_clip:
        utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=dec_grad_clip)
    self.action_dec_optim.step()

    return {
        'decoder_loss':      cost.item(),
        'decoder_grad_norm': grad_norm.item(),
        'value_mean':        value.mean().item(),
        'saturation':        saturation,
        'z_norm':            z_norm,
        'u_norm':            u_norm,
        'hidden_norm':       self.model._action_decoder._hidden_norm,
        'lambda_gae':        lam,
    }
