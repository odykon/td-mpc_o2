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
        (action_decoder, cond_norm):
            action_decoder: nn.Sequential, the decoder (Linear -> ReLU -> Linear).
            cond_norm:      LayerNorm over the conditioning signal only (not
                            `u`), or None when there's no conditioning signal
                            (both flags False). The TOLD encoder's `z` has no
                            output normalisation, so its scale can drift
                            arbitrarily over training; cond_norm decouples the
                            decoder's input scale from that drift.
                            Kept separate from action_decoder rather than
                            attached to it as an attribute — nn.Sequential's
                            forward() just iterates every entry in its
                            _modules dict, so an nn.Module attribute assigned
                            onto it becomes an extra "layer" silently run on
                            the decoder's own output, not on cond.
    """
    if use_raw_obs:
        cond_dim = cfg.obs_shape[0]
    elif use_latent_state:
        cond_dim = cfg.latent_dim
    else:
        cond_dim = 0

    action_decoder = nn.Sequential(
        nn.Linear(cfg.latent_action_dim + cond_dim, 256),
        nn.ReLU(),
        nn.Linear(256, cfg.horizon * cfg.action_dim),
    )
    action_decoder._hidden_norm = 0.0
    cond_norm = nn.LayerNorm(cond_dim) if cond_dim > 0 else None

    def _hidden_norm_hook(module, input, output):
        action_decoder._hidden_norm = output.norm(dim=-1).mean().item()

    action_decoder[1].register_forward_hook(_hidden_norm_hook)

    return action_decoder, cond_norm


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
        cond      = self._cond_norm(cond)
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
    modules = [self._action_decoder]
    if self._cond_norm is not None:
        modules.append(self._cond_norm)
    for m in modules:
        h.set_requires_grad(m, enable)
        if not enable:
            # O2 params are not in self.optim, so optim.zero_grad() never clears them.
            # Zeroing here prevents stale decoder gradients from inflating
            # clip_grad_norm_ during TOLD updates and causing over-clipping.
            for p in m.parameters():
                p.grad = None
