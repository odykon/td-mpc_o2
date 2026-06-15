"""
action_decoder.py
-----------------
Action decoder network for latent action space control.

Provides:
    build_action_decoder — constructs the decoder nn.Module (no Tanh output layer)
    decode_sequence      — method attached to TOLD instances; applies tanh externally
    track_TOLD_grad / track_O2_grad — gradient enable/disable helpers
"""

import torch
import torch.nn as nn
import algorithm.helper as h


def build_action_decoder(cfg, use_latent_state=True, use_raw_obs=False):
    """
    Build the action decoder network.

    Output layer has no activation — tanh is applied externally in decode_sequence.

    Conditioning modes (mutually exclusive, use_raw_obs takes priority):
        use_raw_obs=True:      input is [u, obs]  — raw observation
        use_latent_state=True: input is [u, z]    — encoded latent state
        both False:            input is u only

    Args:
        cfg:              Config with latent_action_dim, latent_dim,
                          action_dim, horizon, obs_shape.
        use_latent_state: Concatenate encoded latent state z to u.
        use_raw_obs:      Concatenate raw observation to u (overrides use_latent_state).

    Returns:
        nn.Sequential: the action decoder (Linear -> ReLU -> Linear).
    """
    if use_raw_obs:
        input_dim = cfg.latent_action_dim + cfg.obs_shape[0]
    elif use_latent_state:
        input_dim = cfg.latent_action_dim + cfg.latent_dim
    else:
        input_dim = cfg.latent_action_dim

    action_decoder = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, cfg.horizon * cfg.action_dim),
    )
    action_decoder._hidden_norm = 0.0

    def _hidden_norm_hook(module, input, output):
        action_decoder._hidden_norm = output.norm(dim=-1).mean().item()

    action_decoder[1].register_forward_hook(_hidden_norm_hook)

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
    nn.init.zeros_(V_net[-1].weight)
    nn.init.zeros_(V_net[-1].bias)
    return V_net


def decode_sequence(self, u, cond, return_pretanh=False):
    """
    Decode a latent action u into an action sequence.

    Tanh is applied externally here (not inside the network).
    Attached to TOLD instances via types.MethodType in TDMPC_O2.__init__.

    Args:
        u:              [B, latent_action_dim] latent actions.
        cond:           [B, cond_dim] conditioning signal — either encoded
                        latent state z or raw observation depending on cfg.
        return_pretanh: if True, also return pre-tanh values.

    Returns:
        actions: [horizon, B, action_dim]
        pretanh: [horizon, B, action_dim]  (only if return_pretanh=True)
    """
    B = u.size(0)
    use_raw_obs = getattr(self.cfg, 'use_raw_obs', False)
    if use_raw_obs or self.cfg.use_latent_state:
        dec_input = torch.cat([u, cond], dim=-1)
    else:
        dec_input = u

    x       = self._action_decoder(dec_input)
    pretanh = x.view(B, self.cfg.horizon, self.cfg.action_dim).permute(1, 0, 2)
    actions = torch.tanh(pretanh)

    if return_pretanh:
        return actions, pretanh
    return actions


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
