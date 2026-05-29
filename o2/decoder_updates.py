"""
decoder_updates.py
------------------
All decoder update strategies for the latent action space.

Provides three update methods, each designed to be bound as a method on
TDMPC_O2 (self = TDMPC_O2 instance):

    action_decoder_DDPG_update    — Off-policy, DDPG-style value maximisation
    action_decoder_DDPG_update_v2 — Same + entropy regularization + saturation penalty
    PG_withV                      — On-policy policy gradient with value baseline
    action_decoder_PPO            — PPO-KL style update (on-policy)

Shared helpers:

    V_net_update    — TD(0) update for the value baseline network
    action_entropy_loss — SAC-style entropy regulariser
    saturation_loss — Tanh log-jacobian penalty (discourages action saturation)

Saturation penalty (action_decoder_DDPG_update_v2):

    Jacobian penalty (inline, fast):
        Computed directly from pretanh values already obtained via
        decode_sequence_pretanh. No extra sampling needed.
        jacobian_penalty = -log(1 - tanh(pretanh)^2 + eps).sum(-1).mean(0)  [B]
        cost = -(value - coeff * jacobian_penalty).mean()
"""

import torch
from torch.distributions import MultivariateNormal
import torch.nn.utils as utils


# ---------------------------------------------------------------------------
# 1. DDPG-style decoder update (off-policy)
# ---------------------------------------------------------------------------

def action_decoder_DDPG_update(self, obs, u_mean, horizon):
    """
    One DDPG-style decoder update step.

    Encodes obs → z (detached), decodes u_mean → action sequence, estimates
    value with gradient flow, backprops -value through the decoder.

    Args:
        obs:     [B, obs_dim] observation batch from the replay buffer.
        u_mean:  [B, latent_action_dim] differentiable latent action mean
                 obtained from DCEMethod(update_mode=True).
        horizon: int planning horizon.

    Returns:
        Scalar decoder loss (for logging).
    """
    self.action_dec_optim.zero_grad()

    z        = self.model.h(obs).detach()
    sequence = self.model.decode_sequence(u_mean, z)
    value    = self.estimate_value_with_grad(z, sequence, horizon).nan_to_num(0)
    cost     = -value.mean()

    cost.backward()
    grad_norm = utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=1)
    self.action_dec_optim.step()

    return cost.item(), grad_norm.item()


# ---------------------------------------------------------------------------
# 1b. DDPG update v2 — with entropy regularization + saturation penalty
# ---------------------------------------------------------------------------

def action_decoder_DDPG_update_v2(self, obs, u_mean, horizon, weights=None, log_det_loss=None):
    """
    DDPG-style decoder update with entropy regularization and saturation penalty.

    Two saturation penalty options are available — select by uncommenting:
      Option 1 (inline jacobian penalty): fast, uses pretanh values directly.
      Option 2 (saturation_loss):         sampled, captures full distribution.

    Args:
        obs:     [B, obs_dim] observation batch from the replay buffer.
        u_mean:  [B, latent_action_dim] differentiable latent action mean
                 obtained from DCEMethod(update_mode=True).
        horizon: int planning horizon.

    Returns:
        dict with keys: decoder_loss, decoder_grad_norm, value_mean,
        jacobian_penalty, saturation, z_norm, u_norm
    """
    self.action_dec_optim.zero_grad()

    z                 = self.model.h(obs).detach()
    z_norm            = z.norm(dim=-1).mean().item()
    u_norm            = u_mean.norm(dim=-1).mean().item()
    sequence, pretanh = self.model.decode_sequence_pretanh(u_mean, z)
    value             = self.estimate_value_with_grad(z, sequence, horizon).nan_to_num(0).squeeze(-1)
    saturation        = pretanh.abs().mean().item()

    # --- Option 1: inline jacobian penalty [B], subtracted per batch element ---
    jacobian_penalty  = -torch.log(1 - sequence.pow(2) + 1e-6).sum(-1).mean(0)  # [B]
    per_sample_cost   = -(value - self.cfg.saturation_coeff * jacobian_penalty)
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
        'jacobian_penalty':  jacobian_penalty.mean().item(),
        'saturation':        saturation,
        'z_norm':            z_norm,
        'u_norm':            u_norm,
    }


# ---------------------------------------------------------------------------
# 1b-debug. DDPG_v2 with per-timestep action gradient tracking
# ---------------------------------------------------------------------------

def action_decoder_DDPG_debug(self, obs, u_mean, horizon, weights=None, log_det_loss=None):
    """
    Identical to action_decoder_DDPG_update_v2 but retains the gradient on
    the decoded sequence to measure how much gradient signal each timestep
    receives. Returns an extra 'action_grad_norms' key: a list of `horizon`
    floats, one per timestep (mean L2 norm over the batch).
    """
    self.action_dec_optim.zero_grad()

    z                 = self.model.h(obs).detach()
    z_norm            = z.norm(dim=-1).mean().item()
    u_norm            = u_mean.norm(dim=-1).mean().item()
    sequence, pretanh = self.model.decode_sequence_pretanh(u_mean, z)
    sequence.retain_grad()
    value             = self.estimate_value_with_grad(z, sequence, horizon).nan_to_num(0).squeeze(-1)
    saturation        = pretanh.abs().mean().item()

    jacobian_penalty  = -torch.log(1 - sequence.pow(2) + 1e-6).sum(-1).mean(0)  # [B]
    per_sample_cost   = -(value - self.cfg.saturation_coeff * jacobian_penalty)
    if weights is not None:
        cost = (per_sample_cost * weights).mean()
    else:
        cost = per_sample_cost.mean()

    diversity_coeff = getattr(self.cfg, 'diversity_coeff', 0.0)
    if log_det_loss is not None and diversity_coeff > 0:
        cost = cost + diversity_coeff * log_det_loss

    cost.backward()

    # sequence.grad: [horizon, B, action_dim] — norm over action_dim, mean over batch
    action_grad_norms = sequence.grad.norm(dim=-1).mean(dim=1).tolist()  # [horizon]

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
        'jacobian_penalty':  jacobian_penalty.mean().item(),
        'saturation':        saturation,
        'z_norm':            z_norm,
        'u_norm':            u_norm,
        'action_grad_norms': action_grad_norms,
    }


# ---------------------------------------------------------------------------
# 1c. Soft-value estimate — log-sum-exp over K fresh trajectory samples
# ---------------------------------------------------------------------------

def soft_value_estimate(self, u_mean, u_std, z, horizon, K, alpha):
    """
    Soft-max value estimate over K trajectories sampled fresh from N(u_mean, u_std).

    Computes:
        V_soft = alpha * log( (1/K) * sum_k exp(V^(k) / alpha) )

    where each V^(k) is the estimated value of a trajectory decoded from
    u^(k) ~ N(u_mean, u_std).  Uses rsample so gradients flow back through
    u_mean and u_std into the decoder.

    Numerically stable via the log-sum-exp trick: shifts by max_V (detached)
    before exponentiating.

    Args:
        u_mean  [B, latent_action_dim]
        u_std   [B, latent_action_dim]
        z       [B, z_dim]  — already encoded and detached
        horizon int
        K       int   — number of fresh trajectory samples
        alpha   float — temperature (→0: hard max, →∞: arithmetic mean)

    Returns:
        soft_val [B]
    """
    B = u_mean.shape[0]

    eps    = torch.randn(K, B, u_mean.shape[-1], device=u_mean.device)
    u_k    = u_mean.unsqueeze(0) + u_std.unsqueeze(0) * eps       # [K, B, latent_dim]
    u_flat = u_k.reshape(K * B, -1)                               # [K*B, latent_dim]
    z_rep  = z.repeat_interleave(K, dim=0)                        # [K*B, z_dim]

    seq  = self.model.decode_sequence(u_flat, z_rep)              # [horizon, K*B, action_dim]
    V    = self.estimate_value_with_grad(z_rep, seq, horizon).nan_to_num(0).squeeze(-1)
    V_k  = V.reshape(K, B)                                        # [K, B]

    max_V    = V_k.detach().max(dim=0).values                     # [B]
    soft_val = max_V + alpha * torch.log(
        torch.exp((V_k - max_V.unsqueeze(0)) / alpha).mean(dim=0)
    )                                                             # [B]
    return soft_val


# ---------------------------------------------------------------------------
# 2. Policy-gradient decoder update with learned value baseline (on-policy)
# ---------------------------------------------------------------------------

def PG_withV(self, obs, u_mean, u_std, reward, obs_t1, original_action,
             next_rewards, next_obses, alpha_v, horizon):
    """
    On-policy PG update for the decoder with a TD(0) value baseline.

    Computes a Monte Carlo return estimate, forms advantages, and updates
    the decoder via REINFORCE weighted by advantage. Also updates the value
    network and applies an entropy bonus.

    Args:
        obs:             [B, obs_dim]
        u_mean:          [B, latent_action_dim] current distribution mean
        u_std:           [B, latent_action_dim] current distribution std
        reward:          [B] rewards at current step
        obs_t1:          [B, obs_dim] next observations
        original_action: [B, latent_action_dim] latent actions from rollout
        next_rewards:    [B, T] multi-step rewards
        next_obses:      [B, T, obs_dim] multi-step next observations
        alpha_v:         float entropy coefficient (from variance_schedule)
        horizon:         int planning horizon

    Returns:
        dict of scalar loss components for logging
    """
    gamma = 0.99

    with torch.no_grad():
        for t in range(1):
            discounted_rewards = next_rewards[:, t] * (gamma ** t)
        final_obs  = next_obses[:, t]
        z_final    = self.model.h(final_obs)
        V_final    = self.model._V(z_final).squeeze(-1)
        Q_estimate = discounted_rewards + (gamma ** (t + 1)) * V_final

    z_0 = self.model.h(obs)
    V_z = self.model._V(z_0).squeeze(-1)

    advantage = Q_estimate - V_z.detach()
    sequence  = self.model.decode_sequence(original_action.detach(), z_0)
    action    = sequence[0, :].squeeze(0)

    x            = torch.cat([z_0, action], dim=-1)
    Reward_loss  = self.model._reward(x)

    latent_dist        = torch.distributions.Normal(loc=u_mean, scale=u_std)
    log_probs_per_dim  = latent_dist.log_prob(original_action)
    current_log_probs  = log_probs_per_dim.mean(dim=1)

    Entropy_loss = self.action_entropy_loss(u_mean, u_std, z_0, num_samples=20)
    V_loss       = self.V_net_update(reward, obs, obs_t1)

    PG_loss      = current_log_probs * advantage
    Decoder_loss = -(PG_loss + self.cfg.dec_reward_coeff * Reward_loss + alpha_v * Entropy_loss).mean()

    self.action_dec_optim.zero_grad()
    Decoder_loss.backward()
    utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=1)
    self.action_dec_optim.step()

    return {
        'Decoder_loss':       Decoder_loss,
        'V_loss':             V_loss,
        'PG_loss':            PG_loss.mean(),
        'Reward_loss':        Reward_loss.mean(),
        'Entropy_loss':       Entropy_loss.mean(),
        'Advantage':          advantage.mean(),
        'current_log_probs':  current_log_probs.mean(),
    }


# ---------------------------------------------------------------------------
# 3. PPO-KL decoder update (on-policy)
# ---------------------------------------------------------------------------

def action_decoder_PPO(self, obs, u_mean, u_std, reward, next_obses,
                       original_log_probs, original_action,
                       original_u_mean, original_u_std,
                       use_kl=False, beta=0.05):
    """
    PPO-style decoder update step.

    Two variants selectable via use_kl:
        use_kl=False — plain policy gradient (REINFORCE)
        use_kl=True  — KL-penalised PG (PPO-KL)

    Args:
        obs:                [B, obs_dim]
        u_mean:             [B, latent_action_dim] current distribution mean
        u_std:              [B, latent_action_dim] current distribution std
        reward:             [T, B] rewards over T steps
        next_obses:         [T, B, obs_dim] next observations (last used for bootstrap)
        original_log_probs: [B] log probs under behaviour policy
        original_action:    [B, latent_action_dim] latent actions from rollout
        original_u_mean:    [B, latent_action_dim] behaviour policy mean
        original_u_std:     [B, latent_action_dim] behaviour policy std
        use_kl:             add KL penalty if True
        beta:               KL penalty coefficient

    Returns:
        Scalar loss (for logging).

    Reference: Schulman et al. (2017). Proximal Policy Optimization Algorithms.
    """
    gamma = self.cfg.discount

    discounted_rewards = 0.0
    current_discount   = 1.0
    for t in range(reward.shape[0]):
        discounted_rewards += current_discount * reward[t]
        current_discount   *= gamma

    with torch.no_grad():
        z_final   = self.model.h(next_obses[-1])
        Q_final   = self.calculate_baselines(z_final, u_mean, u_std, max=True)
        mc_return = (discounted_rewards + current_discount * Q_final).squeeze(-1)

    with torch.no_grad():
        z_0      = self.model.h(obs)
        baseline = self.calculate_baselines(z_0, u_mean, u_std, max=True).squeeze(-1)

    advantage = mc_return - baseline
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    dist_new          = torch.distributions.Normal(loc=u_mean, scale=u_std)
    dist_old          = torch.distributions.Normal(loc=original_u_mean, scale=original_u_std)
    current_log_probs = dist_new.log_prob(original_action).mean(dim=-1)

    PG_loss = current_log_probs * advantage

    if use_kl:
        kl   = torch.distributions.kl.kl_divergence(dist_old, dist_new).sum(dim=-1)
        loss = -(PG_loss - beta * kl).mean()
    else:
        loss = -PG_loss.mean()

    self.action_dec_optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=1)
    self.action_dec_optim.step()

    return loss.item()


# ---------------------------------------------------------------------------
# 4. Value network update — shared by PG and PPO
# ---------------------------------------------------------------------------

def V_net_update(self, reward, obs, next_obs):
    """
    One-step TD(0) update for the value baseline network _V.

    Args:
        reward   (Tensor): [B]
        obs      (Tensor): [B, obs_dim]
        next_obs (Tensor): [B, obs_dim]

    Returns:
        Tensor: scalar MSE loss
    """
    gamma = 0.99

    with torch.no_grad():
        z_next = self.model.h(next_obs)
        V_next = self.model._V(z_next).squeeze(-1)
        target = reward + gamma * V_next

    z_0    = self.model.h(obs)
    V_z    = self.model._V(z_0).squeeze(-1)
    V_loss = (target - V_z).pow(2).mean()

    self.V_optim.zero_grad()
    V_loss.backward()
    self.V_optim.step()

    return V_loss


# ---------------------------------------------------------------------------
# 5. Entropy regulariser — shared by PG and PPO
# ---------------------------------------------------------------------------

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
        Tensor: scalar entropy
    """
    batch      = u_mean.shape[0]
    action_dim = self.cfg.action_dim

    u_dist     = torch.distributions.Normal(u_mean, u_std)
    u_samples  = u_dist.rsample((num_samples,))                   # [S, B, latent_dim]
    u_flat     = u_samples.reshape(-1, u_samples.shape[-1])       # [S*B, latent_dim]
    z_repeated = z_state.repeat_interleave(num_samples, dim=0)    # [S*B, z_dim]

    decoded_seq = self.model.decode_sequence(u_flat, z_repeated)  # [horizon, S*B, action_dim]

    x = (
        decoded_seq
        .permute(1, 0, 2)                                         # [S*B, horizon, action_dim]
        .reshape(num_samples, batch, horizon, action_dim)
        .permute(1, 2, 0, 3)                                      # [B, horizon, S, action_dim]
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


# ---------------------------------------------------------------------------
# 6. Saturation loss — tanh log-jacobian penalty
# ---------------------------------------------------------------------------

def saturation_loss(self, u_mean, u_std, z_state, num_samples=20):
    """
    Tanh log-jacobian penalty across the decoded action distribution.

    Samples latent actions from the CEM distribution, decodes each via the
    decoder, and computes the mean negative log-jacobian of the tanh squashing:
        penalty = -log(1 - tanh(pretanh)^2 + eps)
    This is large when actions are near ±1 (saturated) and zero when pretanh ≈ 0.
    Summed over action_dim and averaged over horizon and samples.

    Args:
        u_mean      (Tensor): [B, latent_dim]
        u_std       (Tensor): [B, latent_dim]
        z_state     (Tensor): [B, z_dim]
        num_samples (int):    Monte-Carlo samples per batch element

    Returns:
        Tensor: [B] per-batch-element penalty
    """
    u_dist     = torch.distributions.Normal(u_mean, u_std)
    u_samples  = u_dist.rsample((num_samples,))                  # [S, B, latent_dim]
    u_flat     = u_samples.reshape(-1, u_samples.shape[-1])      # [S*B, latent_dim]
    z_repeated = z_state.repeat_interleave(num_samples, dim=0)   # [S*B, z_dim]

    _, pretanh   = self.model.decode_sequence_pretanh(u_flat, z_repeated)  # [horizon, S*B, action_dim]
    actions      = torch.tanh(pretanh)
    log_jacobian = torch.log(1 - actions.pow(2) + 1e-6)                   # [horizon, S*B, action_dim]

    # sum over action_dim, reshape to [horizon, S, B], mean over horizon and samples → [B]
    penalty = -log_jacobian.sum(-1)                                        # [horizon, S*B]
    penalty = penalty.view(penalty.shape[0], num_samples, -1)              # [horizon, S, B]
    return penalty.mean(dim=(0, 1))                                        # [B]
