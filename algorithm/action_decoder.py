import torch
import torch.nn as nn
def build_action_decoder(cfg, initialize_fn=None, use_latent_state=True):
    """
    Builds and (optionally) initializes the action decoder network.

    Args:
        cfg: object with attributes:
            - latent_action_dim
            - latent_dim
            - action_dim
            - horizon
        initialize_fn: optional callable for custom initialization
        use_latent_state: if False, input is only latent_action (no latent_state)

    Returns:
        nn.Module: The (optionally initialized) action decoder.
    """
    input_dim = cfg.latent_action_dim + cfg.latent_dim if use_latent_state else cfg.latent_action_dim

    action_decoder = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, cfg.horizon * cfg.action_dim),
        nn.Tanh()
    )
    if initialize_fn is not None:
        action_decoder = initialize_fn(
            action_decoder,
            cfg.latent_action_dim,
            cfg.latent_dim if use_latent_state else 0,
            cfg.action_dim,
            cfg.horizon
        )

    return action_decoder


def action_decoder_DDPG_update(self, obs, u_mean, horizon):
    self.action_dec_optim.zero_grad()
    z= self.model.h(obs)
    z = z.detach()
    
    sequence = self.model.decode_sequence(u_mean, z)  #unsqueeze(0) for no batch version
    value = self.estimate_value(z, sequence, horizon).nan_to_num(0)
    cost = -value.mean()

    cost.backward()
    self.action_dec_optim.step()
    # Replace the _step_count check with:
    return cost.item()


def decode_sequence(self, u, z):
    B = u.size(0)
    in_dim = self.model._action_decoder[0].in_features

    if in_dim == self.cfg.latent_action_dim + self.cfg.latent_dim:
        dec_input = torch.cat([u, z], dim=-1)
    else:
        dec_input= u

    actions = self.model._action_decoder(dec_input)
    return actions.view(B, self.cfg.horizon, self.cfg.action_dim)

def action_decoder_Policy_Gradient(self, obs, u_mean, u_std, reward, next_obses, original_action =None):
    self.action_dec_optim.zero_grad()
    gamma = self.cfg.discount  # or your discount factor

    # Calculate discounted rewards
    discounted_rewards = 0
    current_discount = 1.0
    print("reward shape:", reward.shape)
    for t in range(reward.shape[0]):  # Assuming reward is [T, ...] shaped
        discounted_rewards += current_discount * reward[t]
        current_discount *= gamma

    # Add discounted terminal value
    with torch.no_grad():
        z_final = self.model.h(next_obses[-1])  # Use the last next_obs for terminal value
        Q = torch.min(*self.model.Q(z_final, self.model.pi(z_final, self.cfg.min_std)))
        mc_estimate = discounted_rewards + current_discount * Q
    mc_estimate= mc_estimate.squeeze(-1)

    dist = torch.distributions.Normal(loc=u_mean, scale=u_std)
    if(original_action ==None):
        action = dist.rsample()
        """
        samples = dist.rsample((10,))
        log_probs = dist.log_prob(samples)
        log_probs = log_probs.mean(dim=0)"""
    else:
        action = original_action
    log_probs = dist.log_prob(action)
    log_probs = log_probs.sum(dim=-1)        #mean or sum might both be correct

    """    current_avg_return = mc_estimate.mean().item()
    baseline_alpha = 0.3
    self.cfg.return_baseline = (1 - baseline_alpha) * self.cfg.return_baseline + baseline_alpha * current_avg_return
    print("baseline:", self.cfg.return_baseline)"""

    with torch.no_grad():
        baselines = self.calculate_baselines(self.model.h(obs), u_mean, u_std)
    advantage = mc_estimate - baselines.detach() # - self.cfg.return_baseline
    print("advantage:", advantage.mean())
    print("baselines:", baselines.mean())
    print("log_probs:", log_probs.mean())
    policy_gradient = advantage*log_probs
    print("policy_gradient:", policy_gradient.mean())
    cost = -policy_gradient.mean()
    cost.backward()

    utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=1)
    #torch.nn.utils.clip_grad_value_(self.model._action_decoder.parameters(), clip_value=0.5)
    self.action_dec_optim.step()

    return cost.item()

def action_decoder_PPO(self, obs, u_mean, u_std, reward, next_obses, original_log_probs, original_action, original_u_mean, original_u_std):
    self.action_dec_optim.zero_grad()

    #MC estimate and Advantage Calculation
    gamma = self.cfg.discount  # or your discount factor
    # Calculate discounted rewards
    discounted_rewards = 0
    current_discount = 1.0
    print("reward.shape[0]:", reward.shape[0])
    for t in range(reward.shape[0]):  # Assuming reward is [T, ...] shaped
        discounted_rewards += current_discount * reward[t]
        current_discount *= gamma
    # Add discounted terminal value
    with torch.no_grad():
        #z_final = self.model.h(next_obses[-1])  # Use the last next_obs for terminal value
        #_, u_mean_final, u_std_final, _, _= self.DCEMethod(next_obses[-1], update_mode = True, step = step, t0=False)
        Q = self.calculate_baselines(self.model.h(next_obses[-1]), u_mean_final, u_std_final, max = True)
        #Q = torch.min(*self.model.Q(z_final, self.model.pi(z_final, self.cfg.min_std)))
        mc_estimate = discounted_rewards + current_discount * Q
    mc_estimate= mc_estimate.squeeze(-1)
    print("mc estimate:", mc_estimate.mean())
    print("Q(z_final)", Q.mean())
    with torch.no_grad():
        baselines = self.calculate_baselines(self.model.h(obs), u_mean, u_std, max = True)
    print("baseline:", baselines.mean())
    advantage = mc_estimate - baselines.detach()

    #Current log prob and ratio calculation



    dist_new = torch.distributions.Normal(loc=u_mean, scale=u_std)
    current_log_probs = dist_new.log_prob(original_action)
    current_log_probs = current_log_probs.sum(dim=-1)

    dist_old = torch.distributions.Normal(loc=original_u_mean, scale=original_u_std)
    old_log_probs = dist_old.log_prob(original_action)
    old_log_probs = old_log_probs.sum(dim=-1)

    kl = torch.distributions.kl.kl_divergence(dist_old, dist_new)
    kl = kl.sum(dim=-1)
    print("kl div:", kl.mean())
    #ratio = torch.exp(current_log_probs - old_log_probs)
    #ratio = (current_log_probs - old_log_probs)
    print("   advantage:",advantage.mean())
    #print("   ratio:",ratio.mean())
    print("current log probs:", current_log_probs.mean())
    #surrogate = (ratio * advantage)
    PG_loss = current_log_probs * advantage
    #print("PG loss:", PG_loss.mean())
    beta=0.05
    loss = -(PG_loss.mean())
    #loss = -(PG_loss - beta * kl).mean()
    """loss = -torch.min(
        ratio * advantage,
        torch.clamp(ratio, 0 - clip_ratio, 0 + clip_ratio) * advantage
    ).mean()"""
    print("   loss:",loss)
    loss.backward()
    utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=1)
    self.action_dec_optim.step()

    return loss.item()

from typing import NewType
def PG_withV(self, obs, u_mean, u_std, reward, next_obses, original_log_probs, original_action, original_u_mean, original_u_std,k=6):


    #MC estimate and Advantage Calculation
    gamma = 0.99 #self.cfg.discount  # or your discount factor
    # Calculate discounted rewards
    discounted_rewards = 0
    current_discount = 1.0

    for t in range(k):  # Assuming reward is [T, ...] shaped
        discounted_rewards += current_discount * reward[t]
        current_discount *= gamma

    # Add discounted terminal value


    V_final = self.model_target._V(self.model.h(next_obses[k-1]))

    print("V final:", V_final.mean())

    mc_estimate = discounted_rewards + current_discount * V_final
    mc_estimate= mc_estimate.squeeze(-1)

    V_z = self.model._V(self.model.h(obs))
    print("mc estimate:", mc_estimate.mean())
    print("V_z:", V_z.mean())
    advantage = mc_estimate - V_z.detach()
    #advantage = mc_estimate
    print("   advantage mean:",advantage.mean())


    dist_new = torch.distributions.Independent(
        torch.distributions.Normal(loc=u_mean, scale=u_std), 1
    )
    dist_old = torch.distributions.Independent(
        torch.distributions.Normal(loc=original_u_mean, scale=original_u_std), 1
    )

    neg_entropy = -dist_new.entropy()
    print("   neg_etropy shape:", neg_entropy.shape)

    print("   neg_entropy mean:",neg_entropy.mean())

    new_action = dist_new.rsample()
    current_log_probs = dist_new.log_prob(new_action)
    old_log_probs = dist_old.log_prob(new_action)
    #current_log_probs = dist_new.log_prob(original_action)
    #old_log_probs = dist_old.log_prob(original_action)
    print("   current log probs mean :", current_log_probs.mean())
    print("   old log probs mean  :"   , old_log_probs.mean())

    ratio = torch.exp(current_log_probs - old_log_probs)
    print("   ratio:",ratio.mean())
    kl = torch.distributions.kl.kl_divergence(dist_old, dist_new)
    kl_mean = kl.mean()
    print("   kl div:", kl.mean())



    #target_kl_per_dim = 1e-3  # try 1e-3 … 5e-3
    #target_kl = target_kl_per_dim * self.cfg.latent_action_dim
    """
    with torch.no_grad():
        target_kl = 0.02
        beta = self.cfg.beta
        if kl_mean > 2 * target_kl:
            beta *= 2.0
        elif kl_mean < 0.5 * target_kl:
            beta *= 0.5
        self.cfg.beta = float(torch.clamp(torch.tensor(beta), 1e-5, 100.0))
    """
    """V update"""
    V_loss = 0.5 * (mc_estimate.detach() - V_z).pow(2).mean()
    V_loss = V_loss.mean()
    print("   V_loss:",V_loss)

    self.V_optim.zero_grad()
    V_loss.backward()
    self.V_optim.step()

    """Decoder update"""
    #surrogate = torch.clamp(ratio * advantage)

    PG_loss = neg_entropy * advantage
    dec_loss = -(PG_loss.mean())
    #dec_loss = -(surrogate.mean() - beta * kl_mean)
    #dec_loss = -(surrogate - beta * kl).mean()


    print("   dec_loss:",dec_loss)
    self.action_dec_optim.zero_grad()
    dec_loss.backward()
    utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=1)
    self.action_dec_optim.step()

    return dec_loss.item(), V_loss.item()
