"""
action_decoder.py
-----------------
Action decoder network for latent action space control.

Provides:
    build_action_decoder            — constructs the decoder nn.Module
    initialize_per_horizon_identity — stable weight initialisation
    decode_sequence                 — method attached to TOLD instances
                                      via DCEM_TDMPC.__init__
"""

import torch
import torch.nn as nn
import algorithm.helper as h

def build_action_decoder(cfg, initialize=False, use_latent_state=True):
    """
    Build the action decoder network.

    Args:
        cfg:              Config with latent_action_dim, latent_dim,
                          action_dim, horizon.
        initialize:       If True, apply per-horizon identity initialisation.
        use_latent_state: If True, decoder input is [u, z] concatenated.
                          If False, decoder input is u only.

    Returns:
        nn.Sequential: the action decoder.
    """
    input_dim = (
        cfg.latent_action_dim + cfg.latent_dim
        if use_latent_state
        else cfg.latent_action_dim
    )

    # LayerNorm on first hidden layer (after Linear, before ReLU) — follows Q-function pattern.
    # To revert: remove nn.LayerNorm(256) below.
    action_decoder = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, cfg.horizon * cfg.action_dim),
        nn.Tanh(),
    )

    if initialize:
        action_decoder = initialize_per_horizon_identity(
            action_decoder,
            d_u=cfg.latent_action_dim,
            d_z=cfg.latent_dim if use_latent_state else 0,
            d_a=cfg.action_dim,
            H=cfg.horizon,
        )

    return action_decoder

def build_value_network(latent_dim, mlp_dim):
    """Build value network with zero-initialized output layer."""
    V_net = nn.Sequential(
        nn.Linear(latent_dim, mlp_dim),
        nn.LayerNorm(mlp_dim),
        nn.Tanh(),
        nn.Linear(mlp_dim, mlp_dim),
        nn.ELU(),
        nn.Linear(mlp_dim, 1)
    )
    
    # Zero-initialize last layer
    nn.init.zeros_(V_net[-1].weight)
    nn.init.zeros_(V_net[-1].bias)
    
    return V_net



def initialize_per_horizon_identity(decoder, d_u, d_z, d_a, H):
    """
    Initialise decoder layers so that each horizon step maps directly from
    latent action to action, forming a blockwise near-identity mapping.

    Uses ReLU(x) - ReLU(-x) = x to implement identity through the bottleneck.
    Only the latent action portion of the input is mapped — the latent state
    portion (if present) is zeroed and learned from scratch.

    Args:
        decoder: nn.Sequential (Linear -> ReLU -> Linear -> Tanh)
        d_u:     latent_action_dim
        d_z:     latent_dim (0 if latent state not used)
        d_a:     action_dim
        H:       horizon

    Returns:
        The initialised decoder.
    """
    fc1, relu, fc2, tanh = decoder

    with torch.no_grad():
        nn.init.zeros_(fc1.weight)
        nn.init.zeros_(fc1.bias)
        nn.init.zeros_(fc2.weight)
        nn.init.zeros_(fc2.bias)

        for t in range(H):
            for i in range(d_a):
                latent_idx = t * d_a + i
                if latent_idx >= d_u:
                    break

                h_pos = 2 * latent_idx
                h_neg = 2 * latent_idx + 1

                fc1.weight[h_pos, latent_idx] =  1.0
                fc1.weight[h_neg, latent_idx] = -1.0

                out_idx = t * d_a + i
                fc2.weight[out_idx, h_pos] =  1.0
                fc2.weight[out_idx, h_neg] = -1.0

        fc1.weight += 1e-3 * torch.randn_like(fc1.weight)
        fc2.weight += 1e-3 * torch.randn_like(fc2.weight)

    return decoder


def decode_sequence(self, u, z):
    """
    Decode a latent action u into an action sequence.

    Attached to TOLD instances via types.MethodType in DCEM_TDMPC.__init__.
    'self' here refers to the TOLD model instance.

    Args:
        u: [B, latent_action_dim] latent actions.
        z: [B, latent_dim] latent states.

    Returns:
        actions: [horizon, B, action_dim] decoded action sequence.
    """
    B = u.size(0)
    in_dim = self._action_decoder[0].in_features

    if in_dim == self.cfg.latent_action_dim + self.cfg.latent_dim:
        dec_input = torch.cat([u, z], dim=-1)
    else:
        dec_input = u

    actions = self._action_decoder(dec_input)
    return actions.view(B, self.cfg.horizon, self.cfg.action_dim).permute(1, 0, 2)

def decode_sequence_pretanh(self, u, z):
    """
    Same as decode_sequence but also returns the pre-Tanh values.
    Returns:
        actions:   [horizon, B, action_dim]  post-Tanh
        pretanh:   [horizon, B, action_dim]  pre-Tanh
    """
    B = u.size(0)
    in_dim = self._action_decoder[0].in_features

    if in_dim == self.cfg.latent_action_dim + self.cfg.latent_dim:
        dec_input = torch.cat([u, z], dim=-1)
    else:
        dec_input = u

    # forward through all layers except final Tanh
    x = dec_input
    for layer in self._action_decoder[:-1]:
        x = layer(x)

    # x is pre-Tanh
    actions = torch.tanh(x)

    actions = actions.view(B, self.cfg.horizon, self.cfg.action_dim).permute(1, 0, 2)
    pretanh = x.view(B, self.cfg.horizon, self.cfg.action_dim).permute(1, 0, 2)

    return actions, pretanh
    
def track_TOLD_grad(self, enable=True):
    """Enables/disables gradient tracking of all TOLD components."""
    for m in [self._Q1, self._Q2, self._reward, self._dynamics, self._encoder, self._pi]:
        h.set_requires_grad(m, enable)

def track_O2_grad(self, enable=True):
    for m in [self._action_decoder, self._V]:
        h.set_requires_grad(m, enable)
        if not enable:
            # O2 params are not in self.optim, so optim.zero_grad() never clears them.
            # Zeroing here prevents stale decoder/V gradients from inflating
            # clip_grad_norm_ during TOLD updates and causing over-clipping.
            for p in m.parameters():
                p.grad = None
