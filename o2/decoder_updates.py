"""
decoder_updates.py
------------------
Off-policy DDPG-style decoder update with GAE-weighted multi-horizon value targets.
"""

import torch
import torch.nn.utils as utils
from torch.distributions import MultivariateNormal


def update_decoder_DDPG(self, obs, u_mean, horizon, weights=None, log_det_loss=None, u_std=None):
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
        horizon: int planning horizon (used as GAE rollout length).
        weights: optional [B] importance-sampling weights.

    Returns:
        dict with keys: decoder_loss, decoder_grad_norm, value_mean,
                        saturation, z_norm, u_norm, hidden_norm
    """
    self.action_dec_optim.zero_grad()

    use_raw_obs = getattr(self.cfg, 'use_raw_obs', False)
    z                 = self.model.h(obs).detach()
    cond_dec          = obs if use_raw_obs else self.model_target.h(obs).detach()
    z_norm            = z.norm(dim=-1).mean().item()

    # Reparameterized sample from the DCEM distribution: gradients on the
    # decoder loss flow back through both u_mean and u_std, not just u_mean.
    #if u_std is not None:
    #    u = u_mean + u_std * torch.randn_like(u_mean)
    #else:
    u = u_mean
    u_norm            = u.norm(dim=-1).mean().item()
    sequence, pretanh = self.model.decode_sequence(u, cond_dec, return_pretanh=True)
    saturation        = pretanh.abs().mean().item()

    value = self.estimate_value_GAE(z, sequence, horizon).nan_to_num(0).squeeze(-1)
    lam   = getattr(self.cfg, 'lambda_gae', 0.5)

    if hasattr(self, 'log_alpha_diversity'):
        alpha = self.log_alpha_diversity.exp().item()
    else:
        alpha = getattr(self.cfg, 'diversity_coeff', 0.0)

    diversity_cost = alpha * log_det_loss if (log_det_loss is not None and alpha > 0) else 0.0
    value_cost     = (-value).mean()
    per_sample_cost = -value
    if weights is not None:
        cost = (per_sample_cost * weights).mean()
    else:
        cost = per_sample_cost.mean()
    cost = cost + (diversity_cost if hasattr(diversity_cost, 'backward') else 0.0)

    cost.backward()
    grad_norm = torch.sqrt(sum(
        p.grad.norm() ** 2
        for p in self.model._action_decoder.parameters() if p.grad is not None
    ))
    dec_grad_clip = getattr(self.cfg, 'dec_grad_clip_norm', None)
    if dec_grad_clip:
        utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=dec_grad_clip)
    self.action_dec_optim.step()

    if log_det_loss is not None and hasattr(self, 'log_alpha_diversity'):
        alpha_loss = -(self.log_alpha_diversity * (log_det_loss.detach() + self.log_det_target))
        self.alpha_diversity_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_diversity_optim.step()

    diversity_cost_val = diversity_cost.item() if hasattr(diversity_cost, 'item') else float(diversity_cost)
    return {
        'decoder_loss':      cost.item(),
        'value_cost':        value_cost.item(),
        'diversity_cost':    diversity_cost_val,
        'diversity_alpha':   alpha,
        'decoder_grad_norm': grad_norm.item(),
        'value_mean':        value.mean().item(),
        'saturation':        saturation,
        'z_norm':            z_norm,
        'u_norm':            u_norm,
        'hidden_norm':       self.model._action_decoder._hidden_norm,
        'lambda_gae':        lam,
    }


def PG_withV(self, z_t, z_t1, u_mean, u_std, reward, original_action,
             alpha_v, log_det_loss=None):
    """
    On-policy REINFORCE update for the decoder with a TD(0) value baseline.

    Args:
        z_t:             [B, latent_dim] encoded current obs (pre-computed)
        z_t1:            [B, latent_dim] encoded next obs (pre-computed, shared with V_net_update)
        u_mean:          [B, latent_action_dim] current distribution mean (from DCEM)
        u_std:           [B, latent_action_dim] current distribution std (from DCEM)
        reward:          [B] rewards at current step
        original_action: [B, latent_action_dim] latent actions from rollout
        alpha_v:         float entropy coefficient
        log_det_loss:    differentiable diversity term from DCEM (optional)

    Returns:
        dict of scalar metrics for logging
    """
    gamma = 0.99

    with torch.no_grad():
        q_target  = reward + gamma * self.model._V(z_t1).squeeze(-1)

    advantage = q_target - self.model._V(z_t).squeeze(-1).detach()
    log_probs = torch.distributions.Normal(u_mean, u_std).log_prob(original_action).mean(dim=1)
    entropy   = self.action_entropy_loss(u_mean, u_std, z_t, num_samples=20)
    pg_loss   = log_probs * advantage

    decoder_loss = -(pg_loss + alpha_v * entropy).mean()
    self.action_dec_optim.zero_grad()
    decoder_loss.backward()
    self.action_dec_optim.step()

    return {
        'decoder_loss': decoder_loss,
        'pg_loss':      pg_loss.mean(),
        'entropy':      entropy.mean(),
        'advantage':    advantage.mean(),
        'log_probs':    log_probs.mean(),
    }


def V_net_update(self, reward, z_t, z_t1):
    """
    One-step TD(0) update for the value baseline network _V.

    Args:
        reward  (Tensor): [B]
        z_t     (Tensor): [B, latent_dim] encoded current obs (pre-computed)
        z_t1    (Tensor): [B, latent_dim] encoded next obs (pre-computed, shared with PG_withV)

    Returns:
        Tensor: scalar MSE loss
    """
    gamma = 0.99

    with torch.no_grad():
        target = reward + gamma * self.model._V(z_t1).squeeze(-1)

    v_loss = (target - self.model._V(z_t).squeeze(-1)).pow(2).mean()

    self.V_optim.zero_grad()
    v_loss.backward()
    self.V_optim.step()

    return v_loss


def action_entropy_loss(self, u_mean, u_std, z_state, num_samples=20, horizon=5):
    """
    Entropy of the decoded action distribution (encourages action diversity).

    Samples latent vectors, decodes them, computes empirical covariance, and
    returns the entropy of the resulting multivariate Gaussian.

    Args:
        u_mean      (Tensor): [B, latent_dim]
        u_std       (Tensor): [B, latent_dim]
        z_state     (Tensor): [B, z_dim]
        num_samples (int):    Monte-Carlo samples per batch element
        horizon     (int):    planning horizon

    Returns:
        Tensor: [B] per-batch-element entropy (mean over horizon)
    """
    batch      = u_mean.shape[0]
    action_dim = self.cfg.action_dim

    u_dist     = torch.distributions.Normal(u_mean, u_std)
    u_samples  = u_dist.rsample((num_samples,))                  # [S, B, latent_dim]
    u_flat     = u_samples.reshape(-1, u_samples.shape[-1])      # [S*B, latent_dim]
    z_repeated = z_state.repeat_interleave(num_samples, dim=0)   # [S*B, z_dim]

    decoded_seq = self.model.decode_sequence(u_flat, z_repeated) # [horizon, S*B, action_dim]

    x = (
        decoded_seq
        .permute(1, 0, 2)                                        # [S*B, horizon, action_dim]
        .reshape(num_samples, batch, horizon, action_dim)
        .permute(1, 2, 0, 3)                                     # [B, horizon, S, action_dim]
    )

    x_mean     = x.mean(dim=2, keepdim=True)
    x_centered = x - x_mean
    cov = (1.0 / (num_samples - 1)) * torch.matmul(
        x_centered.transpose(2, 3),
        x_centered,
    )

    B, T, _, D = cov.shape
    mean_flat  = x_mean.squeeze(2).reshape(B * T, D)
    cov_flat   = cov.reshape(B * T, D, D)

    mvn          = MultivariateNormal(loc=mean_flat, covariance_matrix=cov_flat)
    entropy_flat = mvn.entropy()
    entropy      = entropy_flat.view(B, T)

    return entropy.mean(dim=1)  # [B] — mean over horizon, keep batch
