"""
decoder.py
----------
Plain MLP action decoder: [u, z] -> action sequence [H, B, action_dim].
"""

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(self, u_dim, z_dim, horizon, action_dim, hidden=256, num_layers=2):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        layers, d = [], u_dim + z_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, horizon * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, u, z):
        """u: [B, u_dim], z: [B, z_dim] -> actions in [-1, 1], [H, B, action_dim]."""
        x = self.net(torch.cat([u, z], dim=-1))
        return torch.tanh(x).view(-1, self.horizon, self.action_dim).permute(1, 0, 2)
