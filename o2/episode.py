"""
implementation/episode.py

PGEpisode extends the original TDMPC Episode with on-policy fields needed
for policy-gradient decoder updates (u_mean, u_std, latent_action, log_probs).

Usage
-----
Replace every `Episode(cfg, obs)` call in the PG training loop with
`PGEpisode(cfg, obs)`.  The original `Episode` class is unchanged, so
all existing replay-buffer code keeps working without modification.
"""

import torch

# Import the original Episode from wherever it lives in your repo.
# Adjust the path if your project lays it out differently.
from algorithm.helper import Episode   # ← original TDMPC Episode


class PGEpisode(Episode):
    """
    Superset of the original TDMPC Episode that additionally stores
    the latent action sampled by CEM_in_latent at each step.
    """

    def __init__(self, cfg, init_obs):
        super().__init__(cfg, init_obs)
        self.latent_action: list = []

    def add_pg(self, latent_action):
        """Store the latent action produced by CEM_in_latent for the current step."""
        self.latent_action.append(latent_action.squeeze(0))

    def finalize(self):
        if len(self.latent_action) > 0:
            self.latent_action = torch.stack(self.latent_action)

    def sample_batches(self, batch_size=None, shuffle=False):
        """
        Iterate through all transitions in batches.

        Yields 4-tuples: obs_t, reward_t, obs_t1, latent_action_t
        """
        if not isinstance(self.latent_action, torch.Tensor):
            raise RuntimeError("Episode must be finalized before sampling. Call episode.finalize() first.")

        episode_length = len(self)
        obs_t  = self.obs[:-1]
        obs_t1 = self.obs[1:]

        indices = torch.arange(episode_length, device=self.device)
        if shuffle:
            indices = indices[torch.randperm(episode_length)]

        if batch_size is None:
            batch_size = episode_length

        for start in range(0, episode_length, batch_size):
            idx = indices[start: start + batch_size]
            yield obs_t[idx], self.reward[idx], obs_t1[idx], self.latent_action[idx]
