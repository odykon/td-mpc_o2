"""
planning.py
-----------
Planning methods for latent action space control.

    DCEM          — Differentiable CEM with LML soft top-k and gradient tracking.
                    Used for decoder training (update_decoder in training_utils).

    CEM_in_latent — Vanilla CEM with hard top-k elite selection.
                    Non-differentiable. Used for environment interaction.

Both functions take self as their first argument where self is a
TDMPC_O2 instance, giving access to self.model, self.cfg,
self.device, and self.estimate_value.

Dependencies
------------
    lml — Limited Multi-Label Projection (Amos et al., 2019)
"""

import torch

import algorithm.helper as h
from lml import LML
from o2.gmm_diversity import _fit_gmm



def DCEM(self, obs, step=None, sample_final_action=False, use_target=False):
    """
    Differentiable CEM in latent action space with per-iteration gradient tracking.

    Runs under torch.enable_grad(). Registers a hook on u_m at each CEM
    iteration that records the gradient norm flowing back through that
    iteration during backward(). The grad_tracker list is populated only
    after cost.backward() is called by the caller.

    Also computes GMM diversity metrics from the last CEM iteration.

    Args:
        obs:                 [B, obs_dim] or raw numpy observation.
        step:                Current training step (for horizon schedule).
        sample_final_action: If True, sample from the final distribution
                             instead of using the mean.
        use_target:          If True, use the target network for value estimation.

    Returns:
        action:        [action_dim]           first action of planned sequence.
        u_mean:        [B, latent_action_dim] final search distribution mean.
        u_std:         [B, latent_action_dim] final search distribution std.
        latent_action: [B, latent_action_dim] latent action that was decoded.
        log_probs:     scalar                 log prob of latent_action.
        grad_tracker:  list of (iteration, grad_norm) tuples (populated after backward).
        diversity:     dict with action_var and GMM metrics.
        gmm_loss:      differentiable GMM diversity loss.
    """
    obs = obs if isinstance(obs, torch.Tensor) else \
          torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    B = obs.shape[0]
    horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

    grad_tracker = []

    use_raw_obs = getattr(self.cfg, 'use_raw_obs', False)

    with torch.enable_grad():
        z_enc = self.model.h(obs).detach()                                   # [B, z_dim] — CEM rollouts
        z     = z_enc.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1).view(B * self.cfg.latent_num_samples, -1)

        if use_raw_obs:
            cond_dec = obs.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1).view(B * self.cfg.latent_num_samples, -1)
        else:
            z_enc_target = self.model_target.h(obs).detach()                 # [B, z_dim] — decoder input
            cond_dec = z_enc_target.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1).view(B * self.cfg.latent_num_samples, -1)

        u_mean = torch.zeros(B, self.cfg.latent_action_dim,
                             device=self.cfg.device, requires_grad=True)
        u_std  = 2 * torch.ones(B, self.cfg.latent_action_dim,
                                device=self.cfg.device, requires_grad=True)

        for i in range(self.cfg.iterations):
            u_noise   = torch.randn(B, self.cfg.latent_num_samples,
                                    self.cfg.latent_action_dim, device=self.cfg.device)
            u_samples = u_mean.unsqueeze(1) + u_std.unsqueeze(1) * u_noise
            u_flat    = u_samples.view(B * self.cfg.latent_num_samples, self.cfg.latent_action_dim)

            sequence = self.model.decode_sequence(u_flat, cond_dec)

            # Action diversity from last CEM iteration
            if i == self.cfg.iterations - 1:
                N, H, A = self.cfg.latent_num_samples, sequence.shape[0], sequence.shape[-1]
                seq     = sequence.view(H, B, N, A)

                with torch.no_grad():
                    action_var = seq.var(dim=2).mean().item()

                gmm_K       = getattr(self.cfg, 'gmm_K', 2)
                gmm_n_iters = getattr(self.cfg, 'gmm_n_iters', 5)
                log_det_loss, gmm_metrics = _fit_gmm(seq[0:1], K=gmm_K, n_iters=gmm_n_iters)
                diversity = {'action_var': action_var, **gmm_metrics}

            value = self.estimate_value_with_grad(z, sequence, horizon, target=use_target).view(B, self.cfg.latent_num_samples)

            mu     = value.mean(dim=1, keepdim=True).detach()
            sigma  = value.std(dim=1, keepdim=True).detach()

            scores = LML(N=self.cfg.latent_num_elites, verbose=0, eps=1e-4)(
                (value - mu) / (sigma + 1e-5) * self.cfg.lml_temperature
            )

            scores = scores / scores.sum(dim=1, keepdim=True)
            elite_weights = scores.unsqueeze(2)

            u_m = (elite_weights * u_samples).sum(dim=1)
            u_s = torch.sqrt(
                (elite_weights * (u_samples - u_m.unsqueeze(1)) ** 2).sum(dim=1)
                / (scores.sum(dim=1, keepdim=True) + 1e-9)
            )

            def make_hook(iteration):
                def record_grad(grad):
                    grad_tracker.append((iteration, grad.norm().item()))
                return record_grad
            u_m.register_hook(make_hook(i))

            u_mean = u_m
            u_std  = u_s

        with torch.no_grad():
            diversity['u_mean_norm'] = u_mean.norm(dim=-1).mean().item()
            diversity['u_std_mean']  = u_std.mean().item()

        dist  = torch.distributions.Normal(loc=u_mean, scale=u_std)
        latent_action = dist.rsample() if sample_final_action else u_mean
        log_probs     = dist.log_prob(latent_action).squeeze_(0).sum(dim=0)

        cond_final = obs if use_raw_obs else self.model_target.h(obs).detach()
        sequence = self.model.decode_sequence(latent_action, cond_final)
        action   = sequence[0, :].squeeze_(0)

    return action, u_mean, u_std, latent_action, log_probs, grad_tracker, diversity, log_det_loss


def CEM_in_latent(self, obs, step=None, sample_final_action=False):
    """
    Plan using vanilla (non-differentiable) CEM in latent action space.

    Uses hard top-k elite selection. Always runs under torch.no_grad().
    Used for environment interaction during training.

    Args:
        obs:                 Raw observation.
        step:                Current training step (for horizon schedule).
        sample_final_action: Sample from final distribution instead of mean.

    Returns:
        action, u_mean, u_std, latent_action, log_probs
    """
    obs = obs if isinstance(obs, torch.Tensor) else \
          torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    B = obs.shape[0]
    horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

    use_raw_obs = getattr(self.cfg, 'use_raw_obs', False)
    self.std = h.linear_schedule(self.cfg.std_schedule, step)

    with torch.no_grad():
        z     = self.model.h(obs)
        z     = z.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1).view(B * self.cfg.latent_num_samples, -1)

        if use_raw_obs:
            cond_dec = obs.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1).view(B * self.cfg.latent_num_samples, -1)
        else:
            z_target = self.model_target.h(obs)
            cond_dec = z_target.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1).view(B * self.cfg.latent_num_samples, -1)

        u_mean = torch.zeros(self.cfg.latent_action_dim, device=self.cfg.device)
        u_std  = 2 * torch.ones(self.cfg.latent_action_dim, device=self.cfg.device)

        for i in range(self.cfg.iterations):
            u_noise   = torch.randn(self.cfg.latent_num_samples, self.cfg.latent_action_dim,
                                    device=self.cfg.device)
            u_samples = u_mean.unsqueeze(0) + u_std.unsqueeze(0) * u_noise  # [N, d_u]

            sequence = self.model.decode_sequence(u_samples, cond_dec)
            value    = self.estimate_value(z, sequence, horizon).squeeze(1)  # [N]

            elite_idxs    = torch.topk(value, self.cfg.latent_num_elites, dim=0).indices
            elite_samples = u_samples[elite_idxs]

            u_m = elite_samples.mean(dim=0)
            u_s = elite_samples.std(dim=0, unbiased=False).clamp(self.std, 2)

            u_mean = self.cfg.momentum * u_mean + (1 - self.cfg.momentum) * u_m
            u_std  = u_s

        dist  = torch.distributions.Normal(loc=u_mean, scale=u_std)
        latent_action = dist.rsample() if sample_final_action else u_mean
        latent_action = latent_action.unsqueeze(0)

        log_probs  = dist.log_prob(latent_action).squeeze_(0).mean(dim=0)
        cond_final = obs if use_raw_obs else self.model_target.h(obs)
        sequence   = self.model.decode_sequence(latent_action, cond_final)
        action     = sequence[0, :].squeeze_(0)

    return action, u_mean, u_std, latent_action, log_probs
