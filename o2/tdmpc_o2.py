"""
tdmpc_o2.py
-----------
TDMPC_O2: subclass of TDMPC that adds the latent action space (O2) extension.

Adds to TDMPC:
    - Action decoder:  maps latent actions u → action sequences
    - Value network:   V(z) baseline for PG-style updates
    - DCEMethod:       differentiable CEM planner (for decoder training)
    - CEM_in_latent:   standard CEM in latent space (for rollouts)
    - Decoder updates: DDPG, PG, PPO — all in o2/decoder_updates.py
"""

import types
import torch
from copy import deepcopy

from algorithm.tdmpc import TDMPC
from o2.action_decoder import (build_action_decoder, decode_sequence,
                                decode_sequence_pretanh, track_TOLD_grad,
                                track_O2_grad, build_value_network)
from o2.planning import DCEMethod, DCEMethod_v2, DCEMethod_planning, CEM_in_latent
from o2.decoder_updates import (action_decoder_DDPG_update,
                                 action_decoder_DDPG_update_v2,
                                 PG_withV, action_entropy_loss,
                                 V_net_update, saturation_loss)


class TDMPC_O2(TDMPC):
    def __init__(self, cfg):
        super().__init__(cfg)

        decoder = build_action_decoder(
            cfg,
            initialize=cfg.decoder_init,
            use_latent_state=cfg.use_latent_state,
        ).to(self.device)

        self.model._action_decoder        = decoder
        self.model_target._action_decoder = deepcopy(decoder).to(self.device)
        self.action_dec_optim = torch.optim.Adam(
            self.model._action_decoder.parameters(), lr=cfg.lr
        )

        self.model._V        = build_value_network(cfg.latent_dim, cfg.mlp_dim).to(self.device)
        self.model_target._V = deepcopy(self.model._V).to(self.device)
        self.V_optim = torch.optim.Adam(self.model._V.parameters(), lr=cfg.lr)

        for model in [self.model, self.model_target]:
            model.decode_sequence          = types.MethodType(decode_sequence, model)
            model.decode_sequence_pretanh  = types.MethodType(decode_sequence_pretanh, model)
            model.track_TOLD_grad          = types.MethodType(track_TOLD_grad, model)
            model.track_O2_grad            = types.MethodType(track_O2_grad, model)

    def estimate_value_with_grad(self, z, actions, horizon, target=False):
        """estimate_value without @torch.no_grad() — needed for gradient flow in DCEMethod."""
        m = self.model_target if target else self.model
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = m.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*m.Q(z, m.pi(z, self.cfg.min_std)))
        return G

    def DCEMethod(self, *args, **kwargs):
        return DCEMethod(self, *args, **kwargs)

    def DCEMethod_v2(self, *args, **kwargs):
        return DCEMethod_v2(self, *args, **kwargs)

    def DCEMethod_planning(self, *args, **kwargs):
        return DCEMethod_planning(self, *args, **kwargs)

    def CEM_in_latent(self, *args, **kwargs):
        return CEM_in_latent(self, *args, **kwargs)

    def action_decoder_DDPG_update(self, *args, **kwargs):
        return action_decoder_DDPG_update(self, *args, **kwargs)

    def action_decoder_DDPG_update_v2(self, *args, **kwargs):
        return action_decoder_DDPG_update_v2(self, *args, **kwargs)

    def saturation_loss(self, *args, **kwargs):
        return saturation_loss(self, *args, **kwargs)

    def PG_withV(self, *args, **kwargs):
        return PG_withV(self, *args, **kwargs)

    def action_entropy_loss(self, *args, **kwargs):
        return action_entropy_loss(self, *args, **kwargs)

    def V_net_update(self, *args, **kwargs):
        return V_net_update(self, *args, **kwargs)
