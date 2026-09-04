"""
planning.py
-----------
Planning methods for latent action space control.

    DCEM          — Differentiable CEM with LML soft top-k and gradient tracking.
                    Used for decoder training (update_decoder in training_utils).

    CEM_in_latent — Vanilla CEM with hard top-k elite selection.
                    Non-differentiable. Used for environment interaction.
                    Optionally augments the sampled pool at every iteration
                    with trajectories rolled out from the learned policy pi,
                    mirroring TDMPC.plan's pi-trajectory mixture.

All functions take self as their first argument where self is a
TDMPC_O2 instance, giving access to self.model, self.cfg,
self.device, and self.estimate_value.

Dependencies
------------
    lml — Limited Multi-Label Projection (Amos et al., 2019)
"""

import torch

import algorithm.helper as h
from lml import LML


def DCEM(self, obs, step=None, sample_final_action=False, use_target=False):
    """
    Differentiable CEM in latent action space with per-iteration gradient tracking.

    Runs under torch.enable_grad(). Registers a hook on u_m at each CEM
    iteration that records the gradient norm flowing back through that
    iteration during backward() (populated only after the caller calls
    cost.backward()). Also computes action_var/sequence_var diagnostics
    from the last CEM iteration's samples.

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
        diversity:     dict with action_var/sequence_var diagnostics.
    """
    obs = obs if isinstance(obs, torch.Tensor) else \
          torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    B = obs.shape[0]
    horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

    grad_tracker = []

    self.std = h.linear_schedule(self.cfg.std_schedule, step)

    with torch.enable_grad():
        z_enc = self.model.h(obs).detach()                                   # [B, z_dim] — CEM rollouts
        z     = z_enc.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1).view(B * self.cfg.latent_num_samples, -1)

        z_enc_target = self.model_target.h(obs).detach()                     # [B, z_dim] — decoder input
        cond_dec = z_enc_target.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1).view(B * self.cfg.latent_num_samples, -1)

        u_mean = torch.zeros(B, self.cfg.latent_action_dim,
                             device=self.cfg.device, requires_grad=True)
        u_std  = 2 * torch.ones(B, self.cfg.latent_action_dim,
                                device=self.cfg.device, requires_grad=True)

        for i in range(self.cfg.dcem_iterations):
            u_noise   = torch.randn(B, self.cfg.latent_num_samples,
                                    self.cfg.latent_action_dim, device=self.cfg.device)
            u_samples = u_mean.unsqueeze(1) + u_std.unsqueeze(1) * u_noise
            u_flat    = u_samples.view(B * self.cfg.latent_num_samples, self.cfg.latent_action_dim)

            sequence = self.model.decode_sequence(u_flat, cond_dec)

            # Reuse the already-computed sequence, detached, so diversity diagnostics
            # don't backprop into the CEM unroll — avoids a second flow forward pass.
            if i == self.cfg.dcem_iterations - 1:
                diversity_sequence = sequence.detach()
                N, H, A = self.cfg.latent_num_samples, diversity_sequence.shape[0], diversity_sequence.shape[-1]
                seq     = diversity_sequence.view(H, B, N, A)

                with torch.no_grad():
                    action_var   = seq[0].var(dim=1).mean().item()   # first action only: (B,N,A) → var over N
                    sequence_var = seq.var(dim=2).mean().item()       # whole sequence:    (H,B,N,A) → var over N

                diversity = {'action_var': action_var, 'sequence_var': sequence_var}

            value = self.estimate_value_with_grad(z, sequence, horizon, target=use_target).view(B, self.cfg.latent_num_samples)
            median = value.median(dim=1, keepdim=True).values.detach()
            mad    = (value - median).abs().median(dim=1, keepdim=True).values.detach()
            # Straight-through: LML's forward input is median/MAD-normalized and
            # temperature-scaled (controls selection sharpness), but the backward
            # gradient into `value` passes through with coefficient 1 — otherwise
            # lml_temperature and 1/mad would both multiply the Jacobian at every
            # unrolled iteration, compounding into exploding gradients.
            normalized = (value - median) / (mad + 1e-5)
            scaled     = normalized * self.cfg.lml_temperature
            lml_input  = normalized + (scaled - normalized).detach()
            scores = LML(N=self.cfg.latent_num_elites, verbose=0, eps=1e-4)(lml_input)
            scores = scores / scores.sum(dim=1, keepdim=True)
            elite_weights = scores.unsqueeze(2)

            u_m = (elite_weights * u_samples).sum(dim=1)
            u_s = torch.sqrt(
                (elite_weights * (u_samples - u_m.unsqueeze(1)) ** 2).sum(dim=1)
                / (scores.sum(dim=1, keepdim=True) + 1e-9)
            )
            u_s = u_s.clamp(self.std, 2)
            
            def make_hook(iteration):
                def record_grad(grad):
                    grad_tracker.append((iteration, grad.norm().item()))
                return record_grad
            u_m.register_hook(make_hook(i))

            u_mean = self.cfg.momentum * u_mean + (1 - self.cfg.momentum) * u_m
            u_std  = u_s
            

        with torch.no_grad():
            diversity['u_mean_norm'] = u_mean.norm(dim=-1).mean().item()
            diversity['u_std_mean']  = u_std.mean().item()

        dist  = torch.distributions.Normal(loc=u_mean, scale=u_std)
        latent_action = dist.rsample() if sample_final_action else u_mean
        log_probs     = dist.log_prob(latent_action).squeeze_(0).sum(dim=0)

        cond_final = self.model_target.h(obs).detach()
        sequence = self.model.decode_sequence(latent_action, cond_final)
        action   = sequence[0, :].squeeze_(0)

    return action, u_mean, u_std, latent_action, log_probs, grad_tracker, diversity


def CEM_in_latent(self, obs, step=None, sample_final_action=False, t0=True, num_pi_trajs=None):
    """
    Plan using vanilla (non-differentiable) CEM in latent action space.

    Uses hard top-k elite selection. Always runs under torch.no_grad().
    Used for environment interaction during training. At every iteration,
    the sampled pool is augmented with `num_pi_trajs` trajectories rolled
    out from the learned policy pi (mirroring TDMPC.plan's pi mixture),
    inverted through the action decoder into their equivalent u.

    Args:
        obs:                 Raw observation.
        step:                Current training step (for horizon schedule).
        sample_final_action: Sample from final distribution instead of mean.
        t0:                  True at the first step of an episode. When
                             cfg.cem_warmstart is set, the search distribution
                             warm-starts from the previous call's optimized
                             mean (shifted by one timestep) unless t0 — matches
                             the t0 reset semantics of TDMPC.plan's _prev_mean.
        num_pi_trajs:        Number of pi trajectories to mix in each
                             iteration. Defaults to cfg.num_pi_trajs; 0
                             disables the pi-trajectory mixture.

    Returns:
        action, u_mean, u_std, latent_action, log_probs
    """
    if num_pi_trajs is None:
        num_pi_trajs = self.cfg.num_pi_trajs

    obs = obs if isinstance(obs, torch.Tensor) else \
          torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    B = obs.shape[0]
    horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

    self.std = h.linear_schedule(self.cfg.std_schedule, step)
    warmstart = self.cfg.cem_warmstart

    with torch.no_grad():
        z_enc = self.model.h(obs)
        z     = z_enc.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1).view(B * self.cfg.latent_num_samples, -1)

        cond_single = self.model_target.h(obs)
        cond_dec = cond_single.unsqueeze(1).repeat(1, self.cfg.latent_num_samples, 1).view(B * self.cfg.latent_num_samples, -1)

        if num_pi_trajs > 0:
            # Roll out fixed policy trajectories in raw action space, exactly
            # like TDMPC.plan's pi_actions, then invert them into their
            # equivalent latent code u_pi so they can be pooled with the
            # resampled u_samples below. Rolled out for the full cfg.horizon
            # (not the schedule-shrunk `horizon`), since the action decoder's
            # flow is a fixed bijection over cfg.latent_action_dim ==
            # cfg.horizon * action_dim — the schedule only truncates how many
            # steps estimate_value sums over.
            z_pi = z_enc.repeat(num_pi_trajs, 1)
            pi_actions = torch.empty(self.cfg.horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
            for t in range(self.cfg.horizon):
                pi_actions[t] = self.model.pi(z_pi, self.cfg.min_std)
                z_pi, _ = self.model.next(z_pi, pi_actions[t])

            cond_pi    = cond_single.repeat(num_pi_trajs, 1)
            pretanh_pi = torch.atanh(pi_actions.clamp(-1 + 1e-6, 1 - 1e-6))
            u_pi       = self.model.invert_sequence(pretanh_pi, cond_pi)

            z_pool    = torch.cat([z, z_enc.repeat(num_pi_trajs, 1)], dim=0)
            cond_pool = torch.cat([cond_dec, cond_pi], dim=0)
        else:
            z_pool, cond_pool = z, cond_dec

        if warmstart and not t0 and self._prev_pretanh is not None:
            # Shift the previous plan's pretanh sequence by one timestep (the
            # first action was just executed) and invert under the *current*
            # cond — CEM will always decode candidates against cond_single at
            # this step, so the seed must be self-consistent with it, not the
            # cond it was originally produced under.
            shifted = torch.zeros_like(self._prev_pretanh)
            shifted[:-1] = self._prev_pretanh[1:]
            u_mean = self.model.invert_sequence(shifted, cond_single).squeeze(0)
        else:
            u_mean = torch.zeros(self.cfg.latent_action_dim, device=self.cfg.device)
        u_std  = 2 * torch.ones(self.cfg.latent_action_dim, device=self.cfg.device)

        for i in range(self.cfg.iterations):
            u_noise   = torch.randn(self.cfg.latent_num_samples, self.cfg.latent_action_dim,
                                    device=self.cfg.device)
            u_samples = u_mean.unsqueeze(0) + u_std.unsqueeze(0) * u_noise  # [N, d_u]
            u_pool    = torch.cat([u_samples, u_pi], dim=0) if num_pi_trajs > 0 else u_samples

            sequence = self.model.decode_sequence(u_pool, cond_pool)
            value    = self.estimate_value(z_pool, sequence, horizon).squeeze(1)  # [N (+num_pi_trajs)]

            elite_idxs    = torch.topk(value, self.cfg.latent_num_elites, dim=0).indices
            elite_samples = u_pool[elite_idxs]

            u_m = elite_samples.mean(dim=0)
            u_s = elite_samples.std(dim=0, unbiased=False).clamp(self.std, 2)

            u_mean = self.cfg.momentum * u_mean + (1 - self.cfg.momentum) * u_m
            u_std  = u_s

        dist  = torch.distributions.Normal(loc=u_mean, scale=u_std)
        latent_action = dist.rsample() if sample_final_action else u_mean
        latent_action = latent_action.unsqueeze(0)

        log_probs  = dist.log_prob(latent_action).squeeze_(0).mean(dim=0)
        cond_final = cond_single
        sequence   = self.model.decode_sequence(latent_action, cond_final)
        action     = sequence[0, :].squeeze_(0)

        if warmstart:
            _, mean_pretanh = self.model.decode_sequence(
                u_mean.unsqueeze(0), cond_final, return_pretanh=True)
            self._prev_pretanh = mean_pretanh

    return action, u_mean, u_std, latent_action, log_probs


def latent_policy_bootstrap(self, z, target=False):
    """
    One-iteration CEM in latent action space, used as the Q-bootstrap policy
    in place of TD-MPC's deterministic pi(z, min_std) when cfg.latent_bootstrap
    is set (see TDMPC_O2.bootstrap_action).

    Q(z, a) is one-step, so only the first decoded action of each sampled u
    is usable — the rest of the decoded sequence is discarded rather than
    unrolled, matching what pi(z) would have supplied for this same bootstrap
    role (a single action, not a sub-plan).

    Elite selection runs under torch.no_grad() — it doesn't need to be
    differentiable and the sampled u's don't depend on z anyway — but the
    final decode + Q call at the caller is not detached, so gradient still
    flows through z exactly as it did through m.pi(z, min_std).

    Args:
        z:      [B, latent_dim] current latent state to bootstrap from.
        target: use model_target's Q for elite scoring (matches the m the
                caller is using for the surrounding value estimate).

    Returns:
        action: [B, action_dim] — same shape/role as m.pi(z, std) would return.
    """
    m = self.model_target if target else self.model
    B = z.shape[0]
    N, K = self.cfg.bootstrap_num_samples, self.cfg.bootstrap_num_elites

    with torch.no_grad():
        u_samples = 1 * torch.randn(B, N, self.cfg.latent_action_dim, device=self.cfg.device)
        z_rep     = z.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)
        u_flat    = u_samples.view(B * N, self.cfg.latent_action_dim)

        sequence = self.model.decode_sequence(u_flat, z_rep)
        a_flat   = sequence[0]
        value    = torch.min(*m.Q(z_rep, a_flat)).view(B, N)

        elite_idxs = torch.topk(value, K, dim=1).indices
        elite_u    = torch.gather(u_samples, 1, elite_idxs.unsqueeze(-1).expand(-1, -1, self.cfg.latent_action_dim))
        u_mean     = elite_u.mean(dim=1)
        u_final    = u_mean + 0.05 * torch.randn_like(u_mean)

    sequence = self.model.decode_sequence(u_final, z)
    return sequence[0]