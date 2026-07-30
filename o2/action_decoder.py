"""
action_decoder.py
-----------------
Conditional normalizing-flow action decoder for latent action space control.

Provides:
    ConditionalAffineCoupling — one masked affine coupling layer
    ActionFlow            — stack of coupling layers; bijective u <-> pretanh
    build_action_decoder  — constructs the ActionFlow (no Tanh output layer)
    decode_sequence       — method attached to TOLD instances; applies tanh externally
    track_TOLD_grad / track_O2_grad — gradient enable/disable helpers
"""

import torch
import torch.nn as nn
import algorithm.helper as h


class ConditionalAffineCoupling(nn.Module):
    """
    One affine coupling layer (RealNVP-style), masked rather than split: `mask`
    selects the identity part elementwise, so it composes with any `d` (including
    odd) via a fixed checkerboard pattern. Conditioning `cond` feeds the
    conditioner net alongside the masked input but never participates in the
    bijection itself, so it doesn't affect the Jacobian.
    """

    def __init__(self, d, cond_dim, mask, hidden=256):
        super().__init__()
        self.register_buffer('mask', mask)
        self.net = nn.Sequential(
            nn.Linear(d + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * d),
        )

    def _scale_shift(self, x_masked, cond):
        s, t = self.net(torch.cat([x_masked, cond], dim=-1)).chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return s, t

    def forward(self, x, cond):
        x_masked = x * self.mask
        s, t = self._scale_shift(x_masked, cond)
        y = x_masked + (1 - self.mask) * (x * s.exp() + t)
        logdet = s.sum(dim=-1)
        return y, logdet

    def inverse(self, y, cond):
        y_masked = y * self.mask
        s, t = self._scale_shift(y_masked, cond)
        x = y_masked + (1 - self.mask) * ((y - t) * (-s).exp())
        logdet = -s.sum(dim=-1)
        return x, logdet


class ActionFlow(nn.Module):
    """Stack of alternating-mask coupling layers: bijective u <-> pretanh, conditioned on `cond`."""

    def __init__(self, d, cond_dim, hidden=256, num_layers=4):
        super().__init__()
        parity = torch.arange(d) % 2
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(d, cond_dim, (parity == (i % 2)).float(), hidden)
            for i in range(num_layers)
        ])
        self._hidden_norm = 0.0

        def _hidden_norm_hook(module, input, output):
            self._hidden_norm = output.norm(dim=-1).mean().item()

        self.layers[0].net[1].register_forward_hook(_hidden_norm_hook)

    def forward(self, u, cond):
        x, logdet = u, 0.0
        for layer in self.layers:
            x, ld = layer(x, cond)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, x, cond):
        u, logdet = x, 0.0
        for layer in reversed(self.layers):
            u, ld = layer.inverse(u, cond)
            logdet = logdet + ld
        return u, logdet


def build_action_decoder(cfg, use_latent_state=True, use_raw_obs=False):
    """
    Build the action decoder as a conditional normalizing flow.

    `cfg.latent_action_dim` must equal `cfg.horizon * cfg.action_dim` — the
    flow is a bijection, so input and output dimensions must match.

    Conditioning modes (mutually exclusive, use_raw_obs takes priority):
        use_raw_obs=True:      input is [u, obs]  — raw observation
        use_latent_state=True: input is [u, z]    — encoded latent state
        both False:            input is u only

    Args:
        cfg:              Config with latent_action_dim, latent_dim,
                          action_dim, horizon, obs_shape, flow_num_layers,
                          flow_hidden_dim.
        use_latent_state: Concatenate encoded latent state z to u.
        use_raw_obs:      Concatenate raw observation to u (overrides use_latent_state).

    Returns:
        (action_decoder, cond_norm):
            action_decoder: ActionFlow, bijective u <-> pretanh.
            cond_norm:      LayerNorm over the conditioning signal only (not `u`),
                            or None if there's no conditioning signal. Decouples
                            the decoder's input scale from TOLD's unnormalized z.
                            Kept separate from action_decoder (not attached as an
                            attribute) since nn.Module attributes on ActionFlow
                            would register as submodules of the flow itself.
    """
    if use_raw_obs:
        cond_dim = cfg.obs_shape[0]
    elif use_latent_state:
        cond_dim = cfg.latent_dim
    else:
        cond_dim = 0

    d = cfg.latent_action_dim
    hidden = getattr(cfg, 'flow_hidden_dim', 256)
    num_layers = getattr(cfg, 'flow_num_layers', 4)

    action_decoder = ActionFlow(d, cond_dim, hidden=hidden, num_layers=num_layers)
    cond_norm = nn.LayerNorm(cond_dim) if cond_dim > 0 else None

    return action_decoder, cond_norm


def decode_sequence(self, u, cond, return_pretanh=False, return_logdet=False):
    """
    Decode a latent action u into an action sequence via the bijective flow.

    Tanh is applied externally here (not inside the flow).
    Attached to TOLD instances via types.MethodType in TDMPC_O2.__init__.

    Args:
        u:              [B, latent_action_dim] latent actions (flow base noise).
        cond:           [B, cond_dim] conditioning signal — either encoded
                        latent state z or raw observation depending on cfg.
        return_pretanh: if True, also return pre-tanh values.
        return_logdet:  if True, also return the flow's forward log-det
                        Jacobian [B] (u -> pretanh), for exact log-density.

    Returns:
        actions: [horizon, B, action_dim]
        pretanh: [horizon, B, action_dim]  (only if return_pretanh=True)
        logdet:  [B]                        (only if return_logdet=True)
    """
    B = u.size(0)
    use_raw_obs = getattr(self.cfg, 'use_raw_obs', False)
    if use_raw_obs or self.cfg.use_latent_state:
        cond_input = self._cond_norm(cond)
    else:
        cond_input = u.new_zeros(B, 0)

    x, logdet = self._action_decoder(u, cond_input)
    pretanh = x.view(B, self.cfg.horizon, self.cfg.action_dim).permute(1, 0, 2)
    actions = torch.tanh(pretanh)

    if return_pretanh and return_logdet:
        return actions, pretanh, logdet
    if return_pretanh:
        return actions, pretanh
    if return_logdet:
        return actions, logdet
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
