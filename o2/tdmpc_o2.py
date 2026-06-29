"""
tdmpc_o2.py
-----------
TDMPC_O2: subclass of TDMPC that adds the latent action space (O2) extension.

Adds to TDMPC:
    - Action decoder:  maps latent actions u → action sequences
    - DCEM:            differentiable CEM planner (for decoder training)
    - CEM_in_latent:   standard CEM in latent space (for rollouts)
    - CEM_in_latent_open_loop: CEM_in_latent variant that also returns the full
                       decoded sequence + scheduled horizon, for open-loop
                       (plan once, execute a:0..a:H-1, then replan) rollouts
    - Decoder update:  off-policy DDPG with GAE value targets
"""

import math
import types
import torch
from copy import deepcopy

from algorithm.tdmpc import TDMPC
from o2.action_decoder import (build_action_decoder, decode_sequence,
                                track_TOLD_grad, track_O2_grad,
                                build_value_network)
from o2.planning import DCEM, CEM_in_latent, CEM_in_latent_open_loop
from o2.decoder_updates import update_decoder_DDPG, PG_withV, action_entropy_loss, V_net_update


class TDMPC_O2(TDMPC):
    def __init__(self, cfg):
        super().__init__(cfg)

        decoder = build_action_decoder(
            cfg,
            use_latent_state=cfg.use_latent_state,
            use_raw_obs=getattr(cfg, 'use_raw_obs', False),
        ).to(self.device)

        self.model._action_decoder        = decoder
        self.model_target._action_decoder = deepcopy(decoder).to(self.device)
        self.action_dec_optim = torch.optim.Adam(
            self.model._action_decoder.parameters(), lr=cfg.lr
        )

        self.model._V        = build_value_network(cfg.latent_dim, cfg.mlp_dim).to(self.device)
        self.model_target._V = deepcopy(self.model._V).to(self.device)
        self.V_optim = torch.optim.Adam(self.model._V.parameters(), lr=cfg.lr)

        self.log_det_target = getattr(cfg, 'log_det_target', -0.8 * float(cfg.action_dim))
        init_coeff = max(getattr(cfg, 'diversity_coeff', 0.01), 1e-8)
        self.log_alpha_diversity = torch.nn.Parameter(
            torch.tensor([math.log(init_coeff)], device=self.device)
        )
        self.alpha_diversity_optim = torch.optim.Adam([self.log_alpha_diversity], lr=1e-3)

        for model in [self.model, self.model_target]:
            model.decode_sequence = types.MethodType(decode_sequence, model)
            model.track_TOLD_grad = types.MethodType(track_TOLD_grad, model)
            model.track_O2_grad           = types.MethodType(track_O2_grad, model)

    def estimate_value_with_grad(self, z, actions, horizon, target=False):
        """estimate_value without @torch.no_grad() — needed for gradient flow in DCEM."""
        m = self.model_target if target else self.model
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = m.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*m.Q(z, m.pi(z, self.cfg.min_std)))
        return G

    def estimate_value_GAE(self, z, actions, horizon):
        """GAE value estimate in a single forward pass.

        V_λ = Σ_h w_h * V_h, where
            w_h = (1 - λ) * λ^(h-1)  for h < H
            w_H = λ^(H-1)             (absorbs remaining weight)

        O(H) model forward passes vs O(H²) for repeated estimate_value_with_grad calls.
        Produces identical gradients w.r.t. leaf tensors.
        """
        lam     = getattr(self.cfg, 'lambda_gae', 0.5)
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
            V_h          = R + discount * torch.min(*m.Q(z_t, m.pi(z_t, self.cfg.min_std)))
            w_h          = (1 - lam) * lam_pow if t < horizon - 1 else lam_pow
            G_gae        = G_gae + w_h * V_h
            lam_pow     *= lam

        return G_gae

    def DCEM(self, *args, **kwargs):
        return DCEM(self, *args, **kwargs)

    def CEM_in_latent(self, *args, **kwargs):
        return CEM_in_latent(self, *args, **kwargs)

    def CEM_in_latent_open_loop(self, *args, **kwargs):
        return CEM_in_latent_open_loop(self, *args, **kwargs)

    def update_decoder_DDPG(self, *args, **kwargs):
        return update_decoder_DDPG(self, *args, **kwargs)

    def PG_withV(self, *args, **kwargs):
        return PG_withV(self, *args, **kwargs)

    def action_entropy_loss(self, *args, **kwargs):
        return action_entropy_loss(self, *args, **kwargs)

    def V_net_update(self, *args, **kwargs):
        return V_net_update(self, *args, **kwargs)
