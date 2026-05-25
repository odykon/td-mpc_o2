"""
planning.py
-----------
Planning methods for latent action space control.

Provides two functions that are exposed as methods on DCEM_TDMPC:

    DCEMethod     — Differentiable CEM using the LML soft top-k projection.
                    When update_mode=True, gradients flow through the elite
                    selection. Used for decoder training (see training.py).

    CEM_in_latent — Vanilla CEM with hard top-k elite selection.
                    Non-differentiable. Used for environment interaction
                    during training and as an ablation baseline.

Both functions take self as their first argument where self is a
DCEM_TDMPC instance, giving access to self.model, self.cfg,
self.device, and self.estimate_value.

Dependencies
------------
    lml — Limited Multi-Label Projection (Amos et al., 2019)
          pip install git+https://github.com/locuslab/lml.git
"""

import torch

import algorithm.helper as h
from lml import LML

def DCEMethod(self, obs, update_mode=False, step=None, t0=True,
              sample_final_action=False):
    """
    Plan using the Differentiable Cross-Entropy Method in latent action space.

    Args:
        obs:                  Raw observation. Converted to tensor internally
                              when update_mode=False.
        update_mode:          If True, retain computation graph for decoder
                              training. Runs under torch.enable_grad().
                              If False, runs under torch.no_grad().
        step:                 Current training step (for horizon schedule).
        t0:                   Whether this is the first step of an episode.
        sample_final_action:  If True, sample from the final distribution
                              instead of using the mean.
        lml_temperature:      Temperature for the LML soft top-k projection.
                              Higher -> sharper elite selection.

    Returns:
        action:        [action_dim]           first action of planned sequence.
        u_mean:        [B, latent_action_dim] final search distribution mean.
        u_std:         [B, latent_action_dim] final search distribution std.
        latent_action: [B, latent_action_dim] latent action that was decoded.
        log_probs:     scalar                 log prob of latent_action.
    """
    if not update_mode:
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    B = obs.shape[0]
    horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

    context = torch.enable_grad if update_mode else torch.no_grad
    with context():
        # Encode and tile latent state
        z = self.model.h(obs)                                              # [B, latent_dim]
        z = z.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1)
        z = z.view(B * self.cfg.latent_num_samples, -1)                    # [B*N, latent_dim]
        if update_mode:
            z = z.detach()

        # Initialise search distribution
        u_mean = torch.zeros(B, self.cfg.latent_action_dim,
                             device=self.cfg.device, requires_grad=update_mode)
        u_std  = 2 * torch.ones(B, self.cfg.latent_action_dim,
                                device=self.cfg.device, requires_grad=update_mode)

        # CEM iterations
        for i in range(self.cfg.iterations):
            u_noise        = torch.randn(B, self.cfg.latent_num_samples,
                                         self.cfg.latent_action_dim, device=self.cfg.device)
            u_samples      = u_mean.unsqueeze(1) + u_std.unsqueeze(1) * u_noise  # [B, N, d_u]
            u_samples_flat = u_samples.view(B * self.cfg.latent_num_samples, self.cfg.latent_action_dim)

            sequence = self.model.decode_sequence(u_samples_flat, z)
            value    = self.estimate_value_with_grad(z, sequence, horizon).view(B, self.cfg.latent_num_samples)

            # LML soft top-k elite selection
            scores = LML(N=self.cfg.latent_num_elites, verbose=0, eps=1e-4)(value * self.cfg.lml_temperature)
            scores = scores / scores.sum(dim=1, keepdim=True)              # [B, N]

            w   = scores.unsqueeze(2)                                      # [B, N, 1]
            u_m = (w * u_samples).sum(dim=1)                               # [B, d_u]
            u_s = torch.sqrt(
                (w * (u_samples - u_m.unsqueeze(1)) ** 2).sum(dim=1)
                / (scores.sum(dim=1, keepdim=True) + 1e-9)
            ).clamp(self.std, 2)

            u_mean = self.cfg.momentum * u_mean + (1 - self.cfg.momentum) * u_m
            u_std  = u_s

        # Decode final action
        z_0   = self.model.h(obs).detach()
        dist  = torch.distributions.Normal(loc=u_mean, scale=u_std)
        latent_action = dist.rsample() if sample_final_action else u_mean
        log_probs     = dist.log_prob(latent_action).squeeze_(0).sum(dim=0)

        sequence = self.model.decode_sequence(latent_action, z_0)
        action   = sequence[0, :].squeeze_(0)

    return action, u_mean, u_std, latent_action, log_probs


def DCEMethod_v2(self, obs, step=None, t0=True,
                 sample_final_action=False, use_target=False):
    """
    DCEMethod with per-iteration gradient tracking via backward hooks.

    Only runs in update_mode=True. Registers a hook on u_m at each CEM
    iteration that records the gradient norm flowing back through that
    iteration during backward(). The grad_tracker list is returned empty
    and populated only after cost.backward() is called by the caller.

    Returns:
        action, u_mean, u_std, latent_action, log_probs, grad_tracker
        grad_tracker: list of (iteration, grad_norm) tuples, sorted by
                      iteration after backward() completes.
    """
    obs = obs if isinstance(obs, torch.Tensor) else \
          torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    B = obs.shape[0]
    horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

    grad_tracker = []

    with torch.enable_grad():
        z_enc = self.model.h(obs).detach()                                   # [B, z_dim]
        z = z_enc.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1)
        z = z.view(B * self.cfg.latent_num_samples, -1)

        u_mean = torch.zeros(B, self.cfg.latent_action_dim,
                             device=self.cfg.device, requires_grad=True)
        u_std  = 2 * torch.ones(B, self.cfg.latent_action_dim,
                                device=self.cfg.device, requires_grad=True)

        for i in range(self.cfg.iterations):
            u_noise  = torch.randn(B, self.cfg.latent_num_samples,
                                   self.cfg.latent_action_dim, device=self.cfg.device)
            u_samples = u_mean.unsqueeze(1) + u_std.unsqueeze(1) * u_noise
            u_flat    = u_samples.view(B * self.cfg.latent_num_samples, self.cfg.latent_action_dim)

            sequence = self.model.decode_sequence(u_flat, z)
            value    = self.estimate_value_with_grad(z, sequence, horizon, target=use_target).view(B, self.cfg.latent_num_samples)

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

        # Action diversity from last CEM iteration
        N, H, A  = self.cfg.latent_num_samples, sequence.shape[0], sequence.shape[-1]
        seq      = sequence.view(H, B, N, A)
        seq_c    = seq - seq.mean(dim=2, keepdim=True)
        cov      = (seq_c.transpose(-1, -2) @ seq_c) / (N - 1)
        cov      = cov + 1e-6 * torch.eye(A, device=seq.device)
        log_det_loss = -torch.linalg.slogdet(cov)[1].mean()                       # minimise to maximise log-det

        with torch.no_grad():
            action_var     = seq.var(dim=2).mean().item()
            eigvals        = torch.linalg.eigvalsh(cov).clamp(min=0)
            p              = eigvals / (eigvals.sum(dim=-1, keepdim=True) + 1e-8)
            effective_rank = p.mul(-1).mul(torch.log(p + 1e-8)).sum(dim=-1).exp().mean().item()
            diversity      = {'action_var': action_var, 'log_det': -log_det_loss.item(), 'effective_rank': effective_rank}

        dist  = torch.distributions.Normal(loc=u_mean, scale=u_std)
        latent_action = dist.rsample() if sample_final_action else u_mean
        log_probs     = dist.log_prob(latent_action).squeeze_(0).sum(dim=0)

        sequence = self.model.decode_sequence(latent_action, z_enc)
        action   = sequence[0, :].squeeze_(0)

    return action, u_mean, u_std, latent_action, log_probs, grad_tracker, diversity, log_det_loss


def DCEMethod_planning(self, obs, step=None, t0=True, sample_final_action=False):
    """
    Soft-top-k CEM in latent space for environment interaction.

    Same as DCEMethod_v2 but runs under torch.no_grad(), has no gradient hooks,
    and skips diversity computation. Returns the same first five elements as
    DCEMethod_v2 for drop-in use during rollouts.
    """
    obs = obs if isinstance(obs, torch.Tensor) else \
          torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    B = obs.shape[0]
    horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

    with torch.no_grad():
        z_enc = self.model.h(obs)
        z     = z_enc.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1)
        z     = z.view(B * self.cfg.latent_num_samples, -1)

        u_mean = torch.zeros(B, self.cfg.latent_action_dim, device=self.cfg.device)
        u_std  = 2 * torch.ones(B, self.cfg.latent_action_dim, device=self.cfg.device)

        for _ in range(self.cfg.iterations):
            u_noise   = torch.randn(B, self.cfg.latent_num_samples,
                                    self.cfg.latent_action_dim, device=self.cfg.device)
            u_samples = u_mean.unsqueeze(1) + u_std.unsqueeze(1) * u_noise
            u_flat    = u_samples.view(B * self.cfg.latent_num_samples, self.cfg.latent_action_dim)

            sequence = self.model.decode_sequence(u_flat, z)
            value    = self.estimate_value(z, sequence, horizon).view(B, self.cfg.latent_num_samples)

            mu     = value.mean(dim=1, keepdim=True)
            sigma  = value.std(dim=1, keepdim=True)
            scores = LML(N=self.cfg.latent_num_elites, verbose=0, eps=1e-4)(
                (value - mu) / (sigma + 1e-5) * self.cfg.lml_temperature
            )
            scores = scores / scores.sum(dim=1, keepdim=True)
            w      = scores.unsqueeze(2)

            u_m = (w * u_samples).sum(dim=1)
            u_s = torch.sqrt(
                (w * (u_samples - u_m.unsqueeze(1)) ** 2).sum(dim=1)
                / (scores.sum(dim=1, keepdim=True) + 1e-9)
            ).clamp(self.std, 2)

            u_mean = self.cfg.momentum * u_mean + (1 - self.cfg.momentum) * u_m
            u_std  = u_s

        dist          = torch.distributions.Normal(loc=u_mean, scale=u_std)
        latent_action = dist.rsample() if sample_final_action else u_mean
        latent_action = latent_action if latent_action.dim() == 2 else latent_action.unsqueeze(0)
        log_probs     = dist.log_prob(latent_action).squeeze_(0).mean(dim=0)

        sequence = self.model.decode_sequence(latent_action, z_enc)
        action   = sequence[0, :].squeeze_(0)

    return action, u_mean, u_std, latent_action, log_probs


def CEM_in_latent(self, obs, update_mode=False, step=None, t0=True,
                  seed=None, sample_final_action=False, lml_temperature=10):
    """
    Plan using vanilla (non-differentiable) CEM in latent action space.

    Identical interface to DCEMethod. Uses hard top-k elite selection
    instead of LML. Always runs under torch.no_grad().

    Use for environment interaction during training and as the ablation
    baseline in experiments.

    Args:
        obs:                 Raw observation.
        step:                Current training step (for horizon schedule).
        sample_final_action: Sample from final distribution instead of mean.

    Returns:
        Same tuple as DCEMethod.
    """
    if not update_mode:
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    B = obs.shape[0]
    horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

    with torch.no_grad():
        # Encode and tile latent state
        z = self.model.h(obs)
        z = z.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1)
        z = z.view(B * self.cfg.latent_num_samples, -1)

        # Initialise search distribution
        u_mean = torch.zeros(self.cfg.latent_action_dim, device=self.cfg.device)
        u_std  = 2 * torch.ones(self.cfg.latent_action_dim, device=self.cfg.device)

        # CEM iterations
        for i in range(self.cfg.iterations):
            u_noise   = torch.randn(self.cfg.latent_num_samples, self.cfg.latent_action_dim,
                                    device=self.cfg.device)
            u_samples = u_mean.unsqueeze(0) + u_std.unsqueeze(0) * u_noise  # [N, d_u]

            sequence = self.model.decode_sequence(u_samples, z)
            value    = self.estimate_value(z, sequence, horizon).squeeze(1)  # [N]

            # Hard top-k elite selection
            elite_idxs    = torch.topk(value, self.cfg.latent_num_elites, dim=0).indices
            elite_samples = u_samples[elite_idxs]

            u_m = elite_samples.mean(dim=0)
            u_s = elite_samples.std(dim=0, unbiased=False).clamp(self.std, 2)

            u_mean = self.cfg.momentum * u_mean + (1 - self.cfg.momentum) * u_m
            u_std  = u_s

        # Decode final action
        z_0   = self.model.h(obs)
        dist  = torch.distributions.Normal(loc=u_mean, scale=u_std)
        latent_action = dist.rsample() if sample_final_action else u_mean
        latent_action = latent_action.unsqueeze(0)

        log_probs = dist.log_prob(latent_action).squeeze_(0).mean(dim=0)
        sequence  = self.model.decode_sequence(latent_action, z_0)
        action    = sequence[0, :].squeeze_(0)

    return action, u_mean, u_std, latent_action, log_probs
