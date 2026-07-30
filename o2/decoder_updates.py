"""
decoder_updates.py
------------------
Off-policy decoder update loops, GAE-weighted multi-horizon value targets:
    update_decoder_det   — pure DDPG-style, value only, deterministic
    update_decoder_stoch — SAC-style, value + GMM diversity + saturation penalty
"""

import torch
import torch.nn.utils as utils

from o2.gmm_diversity import _fit_gmm_em, _gmm_log_prob_squashed


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

    use_raw_obs = getattr(self.cfg, 'use_raw_obs', False)
    z        = self.model.h(obs).detach()
    cond_dec = obs if use_raw_obs else self.model_target.h(obs).detach()
    z_norm   = z.norm(dim=-1).mean().item()
    u_norm   = u_mean.norm(dim=-1).mean().item()

    sequence = self.model.decode_sequence(u_mean, cond_dec)
    value    = self.estimate_value_GAE(z, sequence, horizon).nan_to_num(0).squeeze(-1)
    lam      = getattr(self.cfg, 'lambda_gae', 0.5)

    value_cost = -value
    cost = (value_cost * weights).mean() if weights is not None else value_cost.mean()

    cost.backward()
    grad_norm = torch.sqrt(sum(
        p.grad.norm() ** 2
        for p in self.model._action_decoder.parameters() if p.grad is not None
    ))
    dec_grad_clip = getattr(self.cfg, 'dec_grad_clip_norm', 10)
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
    SAC-style decoder update using a GAE-weighted sum of n-step TD targets.

    V_λ = Σ_h w_h * V_h, where
        w_h = (1 - λ) * λ^(h-1)  for h < gae_horizons
        w_H = λ^(H-1)             (absorbs remaining weight; all weights sum to 1)

    Config: lambda_gae (default 0.5), gae_horizons (default 5).
    Two independent penalties are added to the value cost:
      - diversity_cost = alpha * log_det_loss, the squashed-GMM log-density
        (_gmm_log_prob_squashed) of the value-optimized sample at step 0 —
        the GMM is fit in pretanh space so it can see saturation directly
        (a fit on the squashed action can't distinguish two points both
        sitting at +-1 the same way pretanh can).
      - saturation_cost = future_sat_coeff * future_jacobian_penalty, the
        tanh-Jacobian term averaged over horizon steps 1..H-1 (independent
        of step 0, which the diversity term already covers) — fixed
        (non-learned) coefficient, does not go through alpha.

    Also fits a GMM (o2/gmm_diversity.py) here, on fresh samples drawn from
    the final u_mean/u_std, rather than in DCEM on the last CEM iteration's
    pre-update samples (see git history for the prior version,
    o2/planning.py). Sample count is cfg.gmm_num_samples, independent of the
    CEM population size (cfg.latent_num_samples). u_mean/u_std are detached
    when building the fit samples (gmm_u) — the fit's final M-step (mu/Sigma)
    still carries gradient back to decoder weights via that decode call (see
    o2/gmm_diversity.py for which parts of EM are frozen vs differentiable),
    but not any further back through u_mean/u_std into DCEM's own unrolled
    CEM iterations. The reparameterized sample used for the value backward
    pass (u = u_mean + u_std*noise) is NOT detached — value_cost and the
    diversity penalty (scored on that same sample against the GMM fit) both
    keep their existing gradient path through u_mean/u_std back into DCEM,
    unchanged.

    Args:
        obs:     [B, obs_dim] observation batch from the replay buffer.
        u_mean:  [B, latent_action_dim] differentiable latent action mean
                 obtained from DCEM().
        horizon: int planning horizon (used as GAE rollout length).
        weights: optional [B] importance-sampling weights.
        u_std:   [B, latent_action_dim] final DCEM search-distribution std,
                 required to sample for the GMM diversity fit.

    Returns:
        dict with keys: decoder_loss, decoder_grad_norm, value_mean,
                        diversity_cost, diversity_alpha, saturation_cost,
                        future_sat_coeff, saturation, z_norm, u_norm,
                        hidden_norm, plus the GMM diversity metrics
                        (gmm_diversity, future_sat_penalty, gmm_pi_balance, ...)
    """
    self.action_dec_optim.zero_grad()

    use_raw_obs = getattr(self.cfg, 'use_raw_obs', False)
    z                 = self.model.h(obs).detach()
    cond_dec          = obs if use_raw_obs else self.model_target.h(obs).detach()
    z_norm            = z.norm(dim=-1).mean().item()

    gmm_num_samples = getattr(self.cfg, 'gmm_num_samples', 512)
    gmm_K           = getattr(self.cfg, 'gmm_K', 2)
    gmm_n_iters     = getattr(self.cfg, 'gmm_n_iters', 7)
    gmm_init        = getattr(self.cfg, 'gmm_init', 'kmeans++')
    gmm_kmeans_iters = getattr(self.cfg, 'gmm_kmeans_iters', 1)

    B         = u_mean.shape[0]
    gmm_noise = torch.randn(B, gmm_num_samples, u_mean.shape[-1], device=u_mean.device)
    gmm_u     = u_mean.detach().unsqueeze(1) + u_std.detach().unsqueeze(1) * gmm_noise
    gmm_u_flat = gmm_u.reshape(B * gmm_num_samples, -1)
    gmm_cond   = cond_dec.unsqueeze(1).repeat(1, gmm_num_samples, 1).reshape(B * gmm_num_samples, -1)

    _, gmm_pretanh = self.model.decode_sequence(gmm_u_flat, gmm_cond, return_pretanh=True)
    H, A = gmm_pretanh.shape[0], gmm_pretanh.shape[-1]
    seq  = gmm_pretanh.view(H, B, gmm_num_samples, A)
    mu_fixed, Sigma_fixed, pi_fixed, gmm_metrics = _fit_gmm_em(
        seq[0:1], K=gmm_K, n_iters=gmm_n_iters, init=gmm_init, kmeans_iters=gmm_kmeans_iters)

    # Reparameterized sample from the DCEM distribution: gradients on the
    # decoder loss flow back through both u_mean and u_std, not just u_mean.

    u = u_mean + u_std * torch.randn_like(u_mean)

    u_norm            = u.norm(dim=-1).mean().item()
    sequence, pretanh = self.model.decode_sequence(u, cond_dec, return_pretanh=True)
    sequence.retain_grad()
    pretanh.retain_grad()
    saturation        = pretanh.abs().mean().item()

    # Diversity penalty — squashed-GMM log-density of the value-optimized
    # sample (first horizon step only, matching what the fit was computed on).
    x_eval       = pretanh[0:1].unsqueeze(2)  # [1, B, 1, A]
    log_det_loss = _gmm_log_prob_squashed(x_eval, mu_fixed, Sigma_fixed, pi_fixed).squeeze(0).squeeze(-1)  # [B]
    gmm_metrics['gmm_diversity'] = -log_det_loss.detach().mean().item()
    alpha = self.log_alpha_diversity.exp().item()
    diversity_cost  = alpha * log_det_loss if alpha > 0 else 0.0

    # Saturation penalty for future steps (1..horizon-1) — independent of the
    # diversity term above (which only covers step 0), fixed (non-learned) coefficient.
    future_sat_coeff = getattr(self.cfg, 'future_sat_coeff', 0.0)
    if horizon > 1:
        future_jacobian_penalty = -torch.log(1 - sequence[1:horizon].pow(2) + 1e-6).sum(dim=-1).mean(dim=0)  # [B]
    else:
        future_jacobian_penalty = torch.zeros_like(log_det_loss)
    saturation_cost = future_sat_coeff * future_jacobian_penalty
    gmm_metrics['future_sat_penalty'] = future_jacobian_penalty.detach().mean().item()

    value = self.estimate_value_GAE(z, sequence, horizon).nan_to_num(0).squeeze(-1)
    lam   = getattr(self.cfg, 'lambda_gae', 0.5)

    value_cost      = (-value).mean()
    per_sample_cost = -value + diversity_cost + saturation_cost
    cost = (per_sample_cost * weights).mean() if weights is not None else per_sample_cost.mean()

    cost.backward()
    grad_norm = torch.sqrt(sum(
        p.grad.norm() ** 2
        for p in self.model._action_decoder.parameters() if p.grad is not None
    ))
    seq_grad_norm     = sequence.grad.norm(dim=-1).mean().item() if sequence.grad is not None else 0.0
    pretanh_grad_norm = pretanh.grad.norm(dim=-1).mean().item() if pretanh.grad is not None else 0.0
    dec_grad_clip = getattr(self.cfg, 'dec_grad_clip_norm', 10)
    if dec_grad_clip:
        utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=dec_grad_clip)
    self.action_dec_optim.step()

    if log_det_loss is not None and hasattr(self, 'log_alpha_diversity'):
        alpha_loss = -(self.log_alpha_diversity * (log_det_loss.detach().mean() + self.log_det_target))
        self.alpha_diversity_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_diversity_optim.step()

    diversity_cost_val = diversity_cost.mean().item() if hasattr(diversity_cost, 'item') else float(diversity_cost)
    return {
        'decoder_loss':      cost.item(),
        'value_cost':        value_cost.item(),
        'diversity_cost':    diversity_cost_val,
        'diversity_alpha':   alpha,
        'decoder_grad_norm': grad_norm.item(),
        'value_mean':        value.mean().item(),
        'saturation':        saturation,
        'saturation_cost':   saturation_cost.mean().item(),
        'future_sat_coeff':  future_sat_coeff,
        'seq_grad_norm':     seq_grad_norm,
        'pretanh_grad_norm': pretanh_grad_norm,
        'z_norm':            z_norm,
        'u_norm':            u_norm,
        'hidden_norm':       self.model._action_decoder._hidden_norm,
        'lambda_gae':        lam,
        **gmm_metrics,
    }