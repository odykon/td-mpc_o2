"""
tdmpc_o2.py
-----------
TDMPC_O2: subclass of TDMPC that adds the latent action space (O2) extension.

Adds to TDMPC:
    - Action decoder:  maps latent actions u → action sequences
    - DCEM:            differentiable CEM planner (for decoder training)
    - CEM_in_latent:   standard CEM in latent space (for rollouts)
    - Decoder update:  off-policy DDPG with GAE value targets
"""

import types
import torch
from copy import deepcopy

from algorithm.tdmpc import TDMPC
from o2.action_decoder import (build_action_decoder, decode_sequence,
                                decode_sequence_pretanh, track_TOLD_grad,
                                track_O2_grad)
from o2.planning import DCEM, CEM_in_latent
from o2.decoder_updates import update_decoder_DDPG


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

        for model in [self.model, self.model_target]:
            model.decode_sequence         = types.MethodType(decode_sequence, model)
            model.decode_sequence_pretanh = types.MethodType(decode_sequence_pretanh, model)
            model.track_TOLD_grad         = types.MethodType(track_TOLD_grad, model)
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

    def DCEM(self, *args, **kwargs):
        return DCEM(self, *args, **kwargs)

    def CEM_in_latent(self, *args, **kwargs):
        return CEM_in_latent(self, *args, **kwargs)

    def update_decoder_DDPG(self, *args, **kwargs):
        return update_decoder_DDPG(self, *args, **kwargs)
