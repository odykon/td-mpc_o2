"""
tdmpc_o2.py
-----------
TDMPC_O2: subclass of TDMPC that adds the latent action space (O2) extension.

Adds to TDMPC:
    - Action decoder:  maps latent actions u → action sequences
    - DCEM:            differentiable CEM planner (for decoder training)
    - CEM_in_latent:   standard CEM in latent space (for rollouts)
    - Decoder update:  off-policy, GAE value targets — DDPG-style (value only)
                       or SAC-style (value + exact flow entropy)
"""

import math
import types
import torch

from algorithm.tdmpc import TDMPC
from o2.action_decoder import (build_action_decoder, decode_sequence, invert_sequence,
                                track_TOLD_grad, track_O2_grad)
from o2.planning import DCEM, CEM_in_latent, latent_policy_bootstrap
from o2.decoder_updates import update_decoder_det, update_decoder_stoch


class TDMPC_O2(TDMPC):
    def __init__(self, cfg):
        super().__init__(cfg)

        decoder, cond_norm = build_action_decoder(
            cfg,
            use_latent_state=cfg.use_latent_state,
        )
        decoder = decoder.to(self.device)
        cond_norm = cond_norm.to(self.device) if cond_norm is not None else None

        self.model._action_decoder = decoder
        self.model._cond_norm      = cond_norm
        # Not attached to model_target: nothing ever reads model_target's decoder
        # (decode_sequence/invert_sequence are only ever called on self.model —
        # conditioning uses model_target.h, the encoder, not its decoder), so
        # giving it a copy just meant h.ema() wastefully Polyak-averaged the
        # flow's weights into it on every TOLD update for no benefit.

        dec_params = list(self.model._action_decoder.parameters())
        if self.model._cond_norm is not None:
            dec_params += list(self.model._cond_norm.parameters())
        self.action_dec_optim = torch.optim.Adam(dec_params, lr=cfg.lr)

        # log_det_target has no fixed default in cfgs/o2_default.yaml since it's
        # computed from horizon/action_dim, which vary per task.
        self.log_det_target = getattr(cfg, 'log_det_target', -float(cfg.horizon * cfg.action_dim))
        init_coeff = max(cfg.diversity_coeff, 1e-8)
        self.log_alpha_diversity = torch.nn.Parameter(
            torch.tensor([math.log(init_coeff)], device=self.device)
        )
        self.alpha_diversity_optim = torch.optim.Adam([self.log_alpha_diversity], lr=1e-3)

        # CEM warm-start: pretanh sequence of the previous CEM_in_latent call's
        # u_mean, shifted by one timestep and re-inverted next step. Reset at
        # episode boundaries (t0) by planning.CEM_in_latent.
        self._prev_pretanh = None

        # Only ever called as agent.model.* (never agent.model_target.*), so
        # these are attached to self.model alone.
        self.model.decode_sequence = types.MethodType(decode_sequence, self.model)
        self.model.invert_sequence = types.MethodType(invert_sequence, self.model)
        self.model.track_TOLD_grad = types.MethodType(track_TOLD_grad, self.model)
        self.model.track_O2_grad   = types.MethodType(track_O2_grad, self.model)

    def bootstrap_action(self, z, m, target=False):
        """Action fed to Q for the value-estimate bootstrap term.

        cfg.latent_bootstrap toggles between TD-MPC's deterministic policy
        (default, unchanged behavior) and one iteration of latent-space CEM
        (latent_policy_bootstrap) scored by the same m's Q.
        """
        if self.cfg.latent_bootstrap:
            return self.latent_policy_bootstrap(z, target=target)
        return m.pi(z, self.cfg.min_std)

    def _td_target(self, next_obs, reward):
        """Overrides TDMPC._td_target: routes the Bellman-target bootstrap
        action through bootstrap_action (same cfg.latent_bootstrap toggle as
        estimate_value_with_grad/estimate_value_GAE).

        target=True: when the toggle is on, latent_policy_bootstrap's elite
        search is scored by model_target.Q, matching the target-Q evaluation
        below — the latent search and the value estimate both use the target
        critic consistently, rather than carrying the base algorithm's
        online-pi/target-Q asymmetry into the latent search too. When the
        toggle is off, bootstrap_action falls through to
        self.model.pi(next_z, min_std), exactly as TDMPC._td_target does —
        this override is then behaviorally identical to the base class.

        Called from TDMPC.update() inside torch.no_grad(), so this comes out
        fully detached automatically — no separate detached path needed.
        """
        next_z = self.model.h(next_obs)
        action = self.bootstrap_action(next_z, self.model, target=True)
        return reward + self.cfg.discount * torch.min(*self.model_target.Q(next_z, action))

    def estimate_value_with_grad(self, z, actions, horizon, target=False):
        """estimate_value without @torch.no_grad() — needed for gradient flow in DCEM."""
        m = self.model_target if target else self.model
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = m.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*m.Q(z, self.bootstrap_action(z, m, target=target)))
        return G

    def estimate_value_GAE(self, z, actions, horizon):
        """GAE value estimate in a single forward pass.

        V_λ = Σ_h w_h * V_h, where
            w_h = (1 - λ) * λ^(h-1)  for h < H
            w_H = λ^(H-1)             (absorbs remaining weight)

        O(H) model forward passes vs O(H²) for repeated estimate_value_with_grad calls.
        Produces identical gradients w.r.t. leaf tensors.
        """
        lam     = self.cfg.lambda_gae
        m       = self.model
        G_gae   = 0
        R       = 0
        discount = 1
        lam_pow  = 1
        z_t      = z

        for t in range(horizon):
            z_t, reward  = m.next(z_t, actions[t])
            R            = R + discount * reward
            discount    *= self.cfg.discount
            V_h          = R + discount * torch.min(*m.Q(z_t, self.bootstrap_action(z_t, m, target=False)))
            w_h          = (1 - lam) * lam_pow if t < horizon - 1 else lam_pow
            G_gae        = G_gae + w_h * V_h
            lam_pow     *= lam

        return G_gae

    def DCEM(self, *args, **kwargs):
        return DCEM(self, *args, **kwargs)

    def CEM_in_latent(self, *args, **kwargs):
        return CEM_in_latent(self, *args, **kwargs)

    def latent_policy_bootstrap(self, *args, **kwargs):
        return latent_policy_bootstrap(self, *args, **kwargs)

    def update_decoder_det(self, *args, **kwargs):
        return update_decoder_det(self, *args, **kwargs)

    def update_decoder_stoch(self, *args, **kwargs):
        return update_decoder_stoch(self, *args, **kwargs)
