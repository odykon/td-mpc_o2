"""
agent.py
--------
TDMPC_Dec: base TDMPC plus an MLP action decoder (separate optimizer,
not part of TOLD / its EMA target).

Extra cfg keys: latent_action_dim, dec_hidden_dim, dec_num_layers,
latent_num_samples, latent_num_elites, dcem_iterations, lml_temperature.
"""

import torch

from algorithm.tdmpc import TDMPC
from o2_base.decoder import MLPDecoder
from o2_base.dcem import dcem
from o2_base.decoder_update import update_decoder


class TDMPC_Dec(TDMPC):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.decoder = MLPDecoder(cfg.latent_action_dim, cfg.latent_dim, cfg.horizon, cfg.action_dim,
                                  hidden=cfg.dec_hidden_dim, num_layers=cfg.dec_num_layers).to(self.device)
        self.dec_optim = torch.optim.Adam(self.decoder.parameters(), lr=cfg.lr)

    def dcem(self, obs, step):
        return dcem(self, obs, step)

    def update_decoder(self, obs, step):
        return update_decoder(self, obs, step)
