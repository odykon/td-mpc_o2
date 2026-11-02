"""
decoder_update.py
-----------------
Decoder update: maximize the value of the decoded final DCEM mean,
backpropagating through the whole DCEM unroll into the decoder.

The gradient has two paths, computed separately so both can be logged:
    direct: through the final decoder(u_mean, z) call only
    unroll: through u_mean, i.e. back through every DCEM iteration and LML
"""

import torch

import algorithm.helper as h
from o2_base.dcem import dcem, estimate_value_with_grad


def grad_norm(grads):
    return torch.cat([g.flatten() for g in grads]).norm().item()


def update_decoder(agent, obs, step):
    """
    Args:
        obs:  [B, obs_dim] observation batch (e.g. from the replay buffer).
        step: global step.

    Returns:
        dict with decoder_loss, value_mean, the reward/Q parts of the value,
        the decoder gradient norms and agent.dcem_stats.
    """
    h.set_requires_grad(agent.model, False)                                 # TOLD is fixed here

    u_mean, _, horizon = dcem(agent, obs, step)
    z = agent.model.h(obs).detach()
    u_final = u_mean.detach().requires_grad_()
    value, reward_part, q_part = estimate_value_with_grad(
        agent, z, agent.decoder(u_final, z), horizon, return_parts=True)
    loss = -value.mean()

    params = list(agent.decoder.parameters())
    *direct, du = torch.autograd.grad(loss, params + [u_final])
    unroll = torch.autograd.grad(u_mean, params, grad_outputs=du)
    total = [d + u for d, u in zip(direct, unroll)]

    agent.dec_optim.zero_grad(set_to_none=True)
    for p, g in zip(params, total):
        p.grad = g
    agent.dec_optim.step()

    h.set_requires_grad(agent.model, True)
    return {
        'decoder_loss': loss.item(),
        'value_mean': value.mean().item(),
        'reward_part': reward_part.mean().item(),
        'q_part': q_part.mean().item(),
        'grad_total': grad_norm(total),
        'grad_direct': grad_norm(direct),
        'grad_unroll': grad_norm(unroll),
        'grad_u_mean': du.norm(dim=-1).mean().item(),
        **agent.dcem_stats,
    }
