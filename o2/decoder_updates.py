"""
decoder_updates.py
------------------
Off-policy DDPG-style decoder update with GAE-weighted multi-horizon value targets.
"""

import torch
import torch.nn.utils as utils


def update_decoder_det(self, obs, u_mean, horizon, weights=None):
    """
    Pure DDPG-style decoder update: optimizes u_mean against the GAE value
    estimate only. No GMM diversity fit, no saturation penalty, no alpha
    tuning — u_mean is decoded deterministically (no reparameterized noise)
    and the value cost is backpropagated straight into the decoder weights.

    Args:
        obs:     [B, obs_dim] observation batch from the replay buffer.
        u_mean:  [B, latent_action_dim] differentiable latent action mean
                 obtained from DCEM().
        horizon: int planning horizon (used as GAE rollout length).
        weights: optional [B] importance-sampling weights.

    Returns:
        dict with keys: decoder_loss, decoder_grad_norm, value_mean,
                        z_norm, u_norm, hidden_norm, lambda_gae
    """
    self.action_dec_optim.zero_grad()

    z        = self.model.h(obs).detach()
    cond_dec = self.model_target.h(obs).detach()
    z_norm   = z.norm(dim=-1).mean().item()
    u_norm   = u_mean.norm(dim=-1).mean().item()

    sequence = self.model.decode_sequence(u_mean, cond_dec)
    value    = self.estimate_value_GAE(z, sequence, horizon).nan_to_num(0).squeeze(-1)
    lam      = self.cfg.lambda_gae

    value_cost = -value
    cost = (value_cost * weights).mean() if weights is not None else value_cost.mean()

    cost.backward()
    grad_norm = torch.sqrt(sum(
        p.grad.norm() ** 2
        for p in self.model._action_decoder.parameters() if p.grad is not None
    ))
    dec_grad_clip = self.cfg.dec_grad_clip_norm
    if dec_grad_clip:
        utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=dec_grad_clip)
    self.action_dec_optim.step()

    return {
        'decoder_loss':      cost.item(),
        'decoder_grad_norm': grad_norm.item(),
        'value_mean':        value.mean().item(),
        'z_norm':            z_norm,
        'u_norm':            u_norm,
        'hidden_norm':       self.model._action_decoder._hidden_norm,
        'lambda_gae':        lam,
    }


def update_decoder_stoch(self, obs, u_mean, horizon, weights=None, u_std=None):
    """
    SAC-style decoder update using a GAE-weighted sum of n-step TD targets,
    with the flow decoder's exact log-density in place of SAC's log-prob term:
    diversity_cost = alpha * log_prob_action, where log_prob_action is the
    exact log-density of the reparameterized sample under the full
    u-space-Gaussian-pushed-through-flow-and-tanh distribution (change of
    variables — no approximation, no fitting). alpha is dual-gradient
    auto-tuned against log_prob_action alone.

    Args:
        obs:     [B, obs_dim] observation batch from the replay buffer.
        u_mean:  [B, latent_action_dim] differentiable latent action mean
                 obtained from DCEM().
        horizon: int planning horizon (used as GAE rollout length).
        weights: optional [B] importance-sampling weights.
        u_std:   [B, latent_action_dim] final DCEM search-distribution std.

    Returns:
        dict with keys: decoder_loss, value_cost, diversity_cost,
                        diversity_alpha, log_prob_action, decoder_grad_norm,
                        value_mean, saturation, saturation_step_0..
                        saturation_step_{H-1} (monitoring only), z_norm,
                        u_norm, hidden_norm, lambda_gae
    """
    self.action_dec_optim.zero_grad()

    z                 = self.model.h(obs).detach()
    cond_dec          = self.model_target.h(obs).detach()
    z_norm            = z.norm(dim=-1).mean().item()

    # Reparameterized sample: gradients flow back through both u_mean and u_std.
    u = u_mean + u_std * torch.randn_like(u_mean)

    u_norm = u.norm(dim=-1).mean().item()
    sequence, pretanh, logdet = self.model.decode_sequence(
        u, cond_dec, return_pretanh=True, return_logdet=True)
    sequence.retain_grad()
    pretanh.retain_grad()
    saturation        = pretanh.abs().mean().item()
    saturation_per_step = {
        f'saturation_step_{t}': pretanh[t].abs().mean().item()
        for t in range(pretanh.shape[0])
    }

    # Exact log-density of the full planned sequence (change of variables through
    # the flow and tanh) — replaces the GMM-EM approximation entirely.
    log_prob_u      = torch.distributions.Normal(u_mean, u_std).log_prob(u).sum(dim=-1)  # [B]
    jacobian_term    = torch.log(1 - sequence.pow(2) + 1e-6).sum(dim=(0, -1))              # [B], full horizon
    log_prob_action = log_prob_u - logdet - jacobian_term                                  # [B]

    alpha = self.log_alpha_diversity.exp().item()
    diversity_cost = alpha * log_prob_action if alpha > 0 else 0.0

    value = self.estimate_value_GAE(z, sequence, horizon).nan_to_num(0).squeeze(-1)
    lam   = self.cfg.lambda_gae

    value_cost      = (-value).mean()
    per_sample_cost = -value + diversity_cost
    cost = (per_sample_cost * weights).mean() if weights is not None else per_sample_cost.mean()

    cost.backward()
    grad_norm = torch.sqrt(sum(
        p.grad.norm() ** 2
        for p in self.model._action_decoder.parameters() if p.grad is not None
    ))
    seq_grad_norm     = sequence.grad.norm(dim=-1).mean().item() if sequence.grad is not None else 0.0
    pretanh_grad_norm = pretanh.grad.norm(dim=-1).mean().item() if pretanh.grad is not None else 0.0
    dec_grad_clip = self.cfg.dec_grad_clip_norm
    if dec_grad_clip:
        utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=dec_grad_clip)
    self.action_dec_optim.step()

    alpha_loss = -(self.log_alpha_diversity * (log_prob_action.detach().mean() + self.log_det_target))
    self.alpha_diversity_optim.zero_grad()
    alpha_loss.backward()
    self.alpha_diversity_optim.step()

    diversity_cost_val = diversity_cost.mean().item() if hasattr(diversity_cost, 'item') else float(diversity_cost)
    return {
        'decoder_loss':      cost.item(),
        'value_cost':        value_cost.item(),
        'diversity_cost':    diversity_cost_val,
        'diversity_alpha':   alpha,
        'log_prob_action':   log_prob_action.detach().mean().item(),
        'decoder_grad_norm': grad_norm.item(),
        'value_mean':        value.mean().item(),
        'saturation':        saturation,
        'seq_grad_norm':     seq_grad_norm,
        'pretanh_grad_norm': pretanh_grad_norm,
        'z_norm':            z_norm,
        'u_norm':            u_norm,
        'hidden_norm':       self.model._action_decoder._hidden_norm,
        'lambda_gae':        lam,
        **saturation_per_step,
    }