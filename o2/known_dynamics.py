"""
known_dynamics.py
------------------
Analytic cartpole dynamics + reward, swapped in for TOLD's learned
`h`/`next` (encoder + dynamics + reward) when cfg.known_dynamics is set
(see TDMPC_O2.__init__). Q, pi, and the action decoder are still trained
normally against real dm_control `cartpole-swingup` transitions — only the
world-model half of TOLD is replaced by the exact analytic model, isolating
decoder/MPC/critic-learning correctness from world-model approximation
error.

Physical state convention: s = [x, xdot, theta, thetadot], theta=0 upright
(unstable equilibrium of the ODE below), theta=pi hanging down — matches
dm_control swingup, which resets hanging down (theta=pi) and rewards
theta=0.
"""

import torch
import torch.nn as nn


class CartpoleDynamics(nn.Module):
    """Batched explicit-Euler cartpole dynamics. s: (..., 4), a: (..., 1) -> (..., 4)."""

    def __init__(self, dt=0.01, n_substeps=4, m_c=1.0, m_p=0.1, l=0.5, g=9.8,
                 force_scale=10.0):
        super().__init__()
        self.dt, self.n_substeps = dt, n_substeps
        self.m_c, self.m_p, self.l, self.g = m_c, m_p, l, g
        self.force_scale = force_scale

    def f(self, s, u):
        x, xdot, th, thdot = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        costh, sinth = torch.cos(th), torch.sin(th)
        temp = (u.squeeze(-1) + self.m_p * self.l * thdot ** 2 * sinth) / (self.m_c + self.m_p)
        thddot = (self.g * sinth - costh * temp) / (
            self.l * (4 / 3 - self.m_p * costh ** 2 / (self.m_c + self.m_p)))
        xddot = temp - self.m_p * self.l * thddot * costh / (self.m_c + self.m_p)
        return torch.stack([xdot, xddot, thdot, thddot], dim=-1)

    def euler_step(self, s, u, dt):
        return s + dt * self.f(s, u)

    def forward(self, s, a):
        u = a * self.force_scale
        sub_dt = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            s = self.euler_step(s, u, sub_dt)
        return s


def known_reward(s, a, x_thresh=2.4, ctrl_cost=0.001):
    """Swing-up shaping reward. An analytic stand-in for dm_control's exact
    `rewards.tolerance`-based reward (not available in closed form) — used
    only to score MPC rollouts. Q/pi/decoder still train against the real
    env reward via the normal TD update, so this need not match exactly."""
    x, th = s[..., 0], s[..., 2]
    upright = (torch.cos(th) + 1) / 2
    centered = torch.exp(-0.5 * (x / x_thresh) ** 2)
    r = upright * centered - ctrl_cost * a.squeeze(-1) ** 2
    return r.unsqueeze(-1)


def physical_h(obs):
    """Fixed (non-learned) recovery of [x, xdot, theta, thetadot] from
    dm_control cartpole's observation [x, cos(theta), sin(theta), xdot,
    thetadot]. Assumes the standard dm_control position+velocity obs
    ordering — verify against env.observation_spec() if this task's obs
    layout differs."""
    x, costh, sinth, xdot, thdot = (
        obs[..., 0], obs[..., 1], obs[..., 2], obs[..., 3], obs[..., 4])
    theta = torch.atan2(sinth, costh)
    return torch.stack([x, xdot, theta, thdot], dim=-1)


def known_h(self, obs):
    """Drop-in replacement for TOLD.h when z is the raw physical state."""
    return physical_h(obs)


def known_next(self, z, a):
    """Drop-in replacement for TOLD.next: z is the physical state directly."""
    return self._known_dynamics(z, a), known_reward(z, a)
