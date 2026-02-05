import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from . import helper as h
from .lml import LML

class TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = h.enc(cfg)
        self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
        self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
        self.apply(h.orthogonal_init)
        for m in [self._reward, self._Q1, self._Q2]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2]:
            h.set_requires_grad(m, enable)

    def track_TOLD_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2, self._reward, self._dynamics, self._encoder]:
            h.set_requires_grad(m, enable)

    def h(self, obs):
        """Encodes an observation into its latent representation (h)."""
        return self._encoder(obs)

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)

    def decode_sequence(self, u, z):
        B = u.size(0)
        in_dim = self._action_decoder[0].in_features

        if in_dim == self.cfg.latent_action_dim + self.cfg.latent_dim:
            dec_input = torch.cat([u, z], dim=-1)
        else:
            dec_input= u

        actions = self._action_decoder(dec_input)
        return actions.view(B, self.cfg.horizon, self.cfg.action_dim).permute(1, 0, 2)  


class TDMPC():
    """Implementation of TD-MPC learning + inference."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)			#was 'cuda'
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.model = TOLD(cfg).to(self.device)
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        self.aug = h.RandomShiftsAug(cfg)
        self.model.eval()
        self.model_target.eval()

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {'model': self.model.state_dict(),
                'model_target': self.model_target.state_dict()}

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)
    
    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d['model'])
        self.model_target.load_state_dict(d['model_target'])

    #@torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
        return G

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation.
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.
        """
        # Seed steps						Commented this to take it out,should put it bakck/ody
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

        # Sample policy trajectories
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
            z = self.model.h(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.model.pi(z, self.cfg.min_std)
                z, _ = self.model.next(z, pi_actions[t])

        # Initialize state and parameters
        z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(self.std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a

    def DCEMethod(self, obs, update_mode=False, step=None, t0=True, seed = None, sample_final_action= False, lml_temperature=10):
        if(not update_mode): obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        B = obs.shape[0]
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

        context= torch.enable_grad if update_mode else torch.no_grad
        with context():
            # Initialize state and parameters
            z = self.model.h(obs)  # shape: [B, latent_dim]
            z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)  # [B, N, latent_dim]
            z = z.view(B * self.cfg.num_samples, -1)
            if update_mode:
                z = z.detach()
                z.requires_grad = True
            u_mean = torch.zeros(B, self.cfg.latent_action_dim, device=self.cfg.device,
                                requires_grad=update_mode)
            u_std = 2 * torch.ones(B, self.cfg.latent_action_dim, device=self.cfg.device, requires_grad = update_mode)

            if(seed!=None): gens = h.sample_u_noise_generators(B, self.cfg.device, seed)
            for i in range(self.cfg.iterations):
                if(seed!=None):
                    noises = []
                    for b in range(B):
                        noise_b = torch.randn(
                            1, self.cfg.num_samples, self.cfg.latent_action_dim,
                            device=self.cfg.device,
                            generator=gens[b]
                        ) 
                        noises.append(noise_b)
                    u_noise = torch.cat(noises, dim=0)  # [B, N, D]
                else: u_noise = torch.randn(B, self.cfg.num_samples, self.cfg.latent_action_dim, device=self.cfg.device)

                u_samples = u_mean.unsqueeze(1) + u_std.unsqueeze(1) * u_noise  # [B, N, latent_action_dim]
                u_samples_flat = u_samples.view(B * self.cfg.num_samples, self.cfg.latent_action_dim)

                sequence = self.model.decode_sequence(u_samples_flat, z)
                value = self.estimate_value(z, sequence, horizon).view(B, self.cfg.num_samples)
                lml_value = value * lml_temperature # temperature parameter
                scores = LML(N=self.cfg.num_elites, verbose=0, eps=1e-4)(lml_value)

                scores = scores / scores.sum(dim=1, keepdim=True)  # [B, N]
                scores_exp = scores.unsqueeze(2)  # [B, N, 1]

                u_m = torch.sum(scores_exp * u_samples, dim=1)  # [B, latent_action_dim]
                u_s = torch.sqrt(torch.sum(scores_exp * (u_samples - u_m.unsqueeze(1)) ** 2, dim=1)) / (scores.sum(dim=1, keepdim=True) + 1e-9)

                u_s = u_s.clamp(self.std, 2)
                u_mean = self.cfg.momentum * u_mean + (1 - self.cfg.momentum) * u_m
                u_std = u_s
            
            z_0 = self.model.h(obs).detach()
            dist = torch.distributions.Normal(loc=u_mean, scale=u_std)
            if(sample_final_action):
                latent_action = dist.rsample()
            else: 
                latent_action = u_mean
            log_probs = dist.log_prob(latent_action).squeeze_(0)
            log_probs = log_probs.sum(dim=0)

            sequence = self.model.decode_sequence(latent_action, z_0)
            action = sequence[0, :].squeeze_(0)
        return action,  u_mean, u_std, latent_action, log_probs
        
    def CEM_on_latent(self, obs, update_mode=False, step=None, t0=True, seed = None, sample_final_action= False, lml_temperature=10):
        if(not update_mode): obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        B = obs.shape[0]
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
    
        with torch.no_grad():
            # Initialize state and parameters
            z = self.model.h(obs)  # shape: [B, latent_dim]
            z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)  # [B, N, latent_dim]
            z = z.view(B * self.cfg.num_samples, -1)
    
            u_mean = torch.zeros(self.cfg.latent_action_dim, device=self.cfg.device,
                                requires_grad=update_mode)
            u_std = 2 * torch.ones(self.cfg.latent_action_dim, device=self.cfg.device, requires_grad = update_mode)
            for i in range(self.cfg.iterations):
                u_noise = torch.randn(self.cfg.num_samples, self.cfg.latent_action_dim, device=self.cfg.device)
                u_samples = u_mean.unsqueeze(0) + u_std.unsqueeze(0) * u_noise  # [B, N, latent_action_dim]
                #u_samples_flat = u_samples.view(self.cfg.num_samples, self.cfg.latent_action_dim)
                #print("u_samples_flat:", u_samples_flat.shape)
    
                sequence = self.model.decode_sequence(u_samples, z)
                value = self.estimate_value(z, sequence, horizon).squeeze(1) #[B, num_samples]
    
                elite_idxs = torch.topk(value, self.cfg.num_elites, dim=0).indices
    
                elite_values = value[elite_idxs]
                elite_samples = u_samples[elite_idxs]
    
                u_m = elite_samples.mean(dim=0)
                u_s  = elite_samples.std(dim=0, unbiased=False)
    
                u_s = u_s.clamp(self.std, 2)
                u_mean = self.cfg.momentum * u_mean + (1 - self.cfg.momentum) * u_m
                u_std = u_s
    
    
            z_0 = self.model.h(obs)
            dist = torch.distributions.Normal(loc=u_mean, scale=u_std)
            if(sample_final_action):
                latent_action = dist.rsample()
            else:
                latent_action = u_mean
            latent_action.unsqueeze_(0)
    
            log_probs = dist.log_prob(latent_action).squeeze_(0)
            log_probs = log_probs.sum(dim=0)
            sequence = self.model.decode_sequence(latent_action, z_0)
            action = sequence[0, :].squeeze_(0)
        return action,  u_mean, u_std, latent_action, log_probs
    
    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0
        for t,z in enumerate(zs):
            a = self.model.pi(z, self.cfg.min_std)
            Q = torch.min(*self.model.Q(z, a))
            pi_loss += -Q.mean() * (self.cfg.rho ** t)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
        self.pi_optim.step()
        self.model.track_q_grad(True)
        return pi_loss.item()
        
    def action_decoder_DDPG_update(self, obs, u_mean, horizon):
        self.action_dec_optim.zero_grad()
        z= self.model.h(obs)
        z = z.detach()
        
        sequence = self.model.decode_sequence(u_mean, z)  #unsqueeze(0) for no batch version
        value = self.estimate_value(z, sequence, horizon).nan_to_num(0)
        cost = -value.mean()
    
        cost.backward()
        self.action_dec_optim.step()
        return cost.item()
    
    @torch.no_grad()
    def _td_target(self, next_obs, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z = self.model.h(next_obs)
        td_target = reward + self.cfg.discount * \
            torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
        return td_target

    def update(self, replay_buffer, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # Representation
        z = self.model.h(self.aug(obs))
        zs = [z.detach()]

        consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        for t in range(self.cfg.horizon):

            # Predictions
            Q1, Q2 = self.model.Q(z, action[t])
            z, reward_pred = self.model.next(z, action[t])
            with torch.no_grad():
                next_obs = self.aug(next_obses[t])
                next_z = self.model_target.h(next_obs)
                td_target = self._td_target(next_obs, reward[t])
            zs.append(z.detach())

            # Losses
            rho = (self.cfg.rho ** t)
            consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
            reward_loss += rho * h.mse(reward_pred, reward[t])
            value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
            priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))

        # Optimize model
        total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
                     self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                     self.cfg.value_coef * value_loss.clamp(max=1e4)
        weighted_loss = (total_loss.squeeze(1) * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
        self.optim.step()
        replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

        # Update policy + target network
        pi_loss = self.update_pi(zs)
        if step % self.cfg.update_freq == 0:
            h.ema(self.model, self.model_target, self.cfg.tau)

        self.model.eval()
        return {'consistency_loss': float(consistency_loss.mean().item()),
                'reward_loss': float(reward_loss.mean().item()),
                'value_loss': float(value_loss.mean().item()),
                'pi_loss': pi_loss,
                'total_loss': float(total_loss.mean().item()),
                'weighted_loss': float(weighted_loss.mean().item()),
                'grad_norm': float(grad_norm)}
