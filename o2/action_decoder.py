"""
action_decoder.py
-----------------
Conditional normalizing-flow action decoder for latent action space control.

Provides:
    ConditionalAffineCoupling — one masked affine coupling layer
    ActionFlow            — stack of coupling layers; bijective u <-> pretanh
    FactoredActionFlow    — first-action flow + rest-of-sequence flow conditioned on a_1
    ActionMLP             — plain MLP alternative (non-invertible, no log-det)
    build_action_decoder  — constructs ActionFlow or ActionMLP per cfg.decoder_type (no Tanh output layer)
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

    def __init__(self, d, cond_dim, mask, hidden=256, zero_init=False, learnable_scale=False):
        super().__init__()
        self.register_buffer('mask', mask)
        self.net = nn.Sequential(
            nn.Linear(d + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * d),
        )
        if zero_init:
            # s = t = 0 for any input -> layer starts as the exact identity
            nn.init.zeros_(self.net[2].weight)
            nn.init.zeros_(self.net[2].bias)
        # per-dim bound on the log-scale, s in [-exp(log_scale), exp(log_scale)]; init 1 = plain tanh
        self.log_scale = nn.Parameter(torch.zeros(d)) if learnable_scale else None

    def _scale_shift(self, x_masked, cond):
        s, t = self.net(torch.cat([x_masked, cond], dim=-1)).chunk(2, dim=-1)
        s = torch.tanh(s)
        if self.log_scale is not None:
            s = s * self.log_scale.exp()
        s = s * (1 - self.mask)
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

    def __init__(self, d, cond_dim, hidden=256, num_layers=4, zero_init=False, learnable_scale=False):
        super().__init__()
        parity = torch.arange(d) % 2
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(d, cond_dim, (parity == (i % 2)).float(), hidden,
                                      zero_init, learnable_scale)
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


class FactoredActionFlow(nn.Module):
    """
    Two-stage autoregressive flow over a flattened [H * action_dim] sequence:
        x_1   = first(u_1, cond)                  — first action, conditioned on state
        x_2:H = rest(u_2:H, [cond, tanh(x_1)])    — remaining actions, conditioned on state + a_1
    x_1 depends only on u_1, so the Jacobian is block-triangular (total log-det is the
    sum of the two flows' log-dets) and the marginal p(a_1 | s) is exact via `first`
    alone. Same forward/inverse interface as ActionFlow.
    """

    def __init__(self, action_dim, horizon, cond_dim, hidden=256, num_layers=4, zero_init=False,
                 learnable_scale=False):
        super().__init__()
        assert horizon >= 2, 'FactoredActionFlow needs horizon >= 2'
        self.action_dim = action_dim
        self.first = ActionFlow(action_dim, cond_dim, hidden, num_layers, zero_init, learnable_scale)
        self.rest = ActionFlow((horizon - 1) * action_dim, cond_dim + action_dim, hidden, num_layers,
                               zero_init, learnable_scale)

    @property
    def _hidden_norm(self):
        return self.first._hidden_norm

    def forward(self, u, cond):
        u1, u_rest = u[:, :self.action_dim], u[:, self.action_dim:]
        x1, ld1 = self.first(u1, cond)
        x_rest, ld_rest = self.rest(u_rest, torch.cat([cond, torch.tanh(x1)], dim=-1))
        return torch.cat([x1, x_rest], dim=-1), ld1 + ld_rest

    def inverse(self, x, cond):
        x1, x_rest = x[:, :self.action_dim], x[:, self.action_dim:]
        u1, ld1 = self.first.inverse(x1, cond)
        u_rest, ld_rest = self.rest.inverse(x_rest, torch.cat([cond, torch.tanh(x1)], dim=-1))
        return torch.cat([u1, u_rest], dim=-1), ld1 + ld_rest

    def log_prob_first(self, a1, cond, u_mean=None, u_std=None, pretanh=None):
        """
        Exact log p(a_1 | s) [B]. Base is N(0, I) unless u_mean/u_std ([B or 1, action_dim],
        e.g. the first action_dim entries of the CEM Gaussian) are given. Pass `pretanh`
        to skip the atanh, which is ill-conditioned near |a1| = 1.
        """
        x1 = pretanh if pretanh is not None else torch.atanh(a1.clamp(-1 + 1e-6, 1 - 1e-6))
        a1 = torch.tanh(x1)
        u1, ld_inv = self.first.inverse(x1, cond)
        u_mean = torch.zeros_like(u1) if u_mean is None else u_mean
        u_std = torch.ones_like(u1) if u_std is None else u_std
        log_prob_u = torch.distributions.Normal(u_mean, u_std).log_prob(u1).sum(dim=-1)
        jacobian_term = torch.log(1 - a1.pow(2) + 1e-6).sum(dim=-1)
        return log_prob_u + ld_inv - jacobian_term

    def act_first(self, cond, u1=None, deterministic=False, return_log_prob=False):
        """
        Policy over the first action only (`rest` is never run). u1 ~ N(0, I) via
        reparameterized sampling unless given; deterministic uses u1 = 0 (image of the
        base mode, not necessarily the mode of p(a_1)).

        Returns a1 [B, action_dim], plus log p(a1 | s) [B] if return_log_prob.
        """
        if u1 is None:
            shape = (cond.size(0), self.action_dim)
            u1 = cond.new_zeros(shape) if deterministic else torch.randn(shape, device=cond.device)
        x1, ld = self.first(u1, cond)
        a1 = torch.tanh(x1)
        if not return_log_prob:
            return a1
        log_prob_u = torch.distributions.Normal(0.0, 1.0).log_prob(u1).sum(dim=-1)
        jacobian_term = torch.log(1 - a1.pow(2) + 1e-6).sum(dim=-1)
        return a1, log_prob_u - ld - jacobian_term


class ActionMLP(nn.Module):
    """
    Plain MLP decoder [u, cond] -> pretanh. Same (x, logdet) forward interface
    as ActionFlow so decode_sequence doesn't need to branch, but logdet is None
    and there is no inverse — not usable with CEM warm-start, pi-trajectory
    mixing, or update_decoder_stoch (all of which need the bijection).
    """

    def __init__(self, u_dim, cond_dim, out_dim, hidden=256, num_layers=4):
        super().__init__()
        layers, d = [], u_dim + cond_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self._hidden_norm = 0.0

        def _hidden_norm_hook(module, input, output):
            self._hidden_norm = output.norm(dim=-1).mean().item()

        self.net[1].register_forward_hook(_hidden_norm_hook)

    def forward(self, u, cond):
        return self.net(torch.cat([u, cond], dim=-1)), None

    def inverse(self, x, cond):
        raise NotImplementedError(
            'ActionMLP is not invertible: disable cem_warmstart / num_pi_trajs with decoder_type=mlp')


def build_action_decoder(cfg, use_latent_state=True):
    """
    Build the action decoder: a conditional normalizing flow
    (cfg.decoder_type='flow', default) or a plain MLP (cfg.decoder_type='mlp').

    flow: `cfg.latent_action_dim` must equal `cfg.horizon * cfg.action_dim` —
          the flow is a bijection, so input and output dimensions must match.
    mlp:  `cfg.latent_action_dim` is free; output is still horizon * action_dim.

    Conditioning modes:
        use_latent_state=True:  input is [u, z]  — encoded latent state
        use_latent_state=False: input is u only

    Args:
        cfg:              Config with decoder_type, latent_action_dim,
                          latent_dim, action_dim, horizon, dec_num_layers,
                          dec_hidden_dim.
        use_latent_state: Concatenate encoded latent state z to u.

    Returns:
        (action_decoder, cond_norm):
            action_decoder: ActionFlow (bijective u <-> pretanh) or ActionMLP.
            cond_norm:      LayerNorm over the conditioning signal only (not `u`),
                            or None if there's no conditioning signal. Decouples
                            the decoder's input scale from TOLD's unnormalized z.
                            Kept separate from action_decoder (not attached as an
                            attribute) since nn.Module attributes on ActionFlow
                            would register as submodules of the flow itself.
    """
    cond_dim = cfg.latent_dim if use_latent_state else 0

    d = cfg.latent_action_dim
    out_dim = cfg.horizon * cfg.action_dim
    hidden = cfg.dec_hidden_dim
    num_layers = cfg.dec_num_layers
    decoder_type = cfg.get('decoder_type', 'flow')

    if decoder_type == 'flow':
        assert d == out_dim, f'flow decoder needs latent_action_dim == horizon * action_dim ({out_dim}), got {d}'
        action_decoder = ActionFlow(d, cond_dim, hidden=hidden, num_layers=num_layers)
    elif decoder_type == 'mlp':
        action_decoder = ActionMLP(d, cond_dim, out_dim, hidden=hidden, num_layers=num_layers)
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type!r} (expected 'flow' or 'mlp')")
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
    if self.cfg.use_latent_state:
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


def invert_sequence(self, pretanh, cond, return_logdet=False):
    """
    Inverse of decode_sequence: map a pretanh action sequence back to latent u.

    Used for CEM warm-starting — a previous plan's pretanh sequence, shifted
    by one timestep, is inverted under the *current* conditioning to seed the
    next CEM search. The conditioning must be the one the caller intends to
    decode `u` back under (the bijection only round-trips for a matched
    forward/inverse cond), which for warm-starting is the new state's cond,
    not the one the original sequence was decoded from.

    Args:
        pretanh: [horizon, B, action_dim] pretanh actions.
        cond:    [B, cond_dim] conditioning signal, same convention as
                 decode_sequence.
        return_logdet: if True, also return the inverse log-det Jacobian.

    Returns:
        u:      [B, latent_action_dim]
        logdet: [B] (only if return_logdet=True)
    """
    horizon, B, action_dim = pretanh.shape
    if self.cfg.use_latent_state:
        cond_input = self._cond_norm(cond)
    else:
        cond_input = pretanh.new_zeros(B, 0)

    x = pretanh.permute(1, 0, 2).reshape(B, horizon * action_dim)
    u, logdet = self._action_decoder.inverse(x, cond_input)

    if return_logdet:
        return u, logdet
    return u


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
