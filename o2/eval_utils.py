import os
import time
import torch
import numpy as np
import imageio

def evaluate_agent(env, agent, cfg, step, n_episodes=5, save_dir=None, video_mode="none", policy="cem_latent", sample_final_action=False):
    """
    Evaluate the agent and optionally save videos.

    Args:
        env:         Environment (DMControl or Gym-like)
        agent:       TDMPC or TDMPC_O2 instance
        cfg:         Config
        step:        Current training step number
        n_episodes:  Number of evaluation episodes
        save_dir:    Directory where videos are saved (optional)
        video_mode:  "first", "best_worst", or "none"
        policy:      Action-selection strategy:
                       "cem_latent"    - CEM search in latent action space (agent.CEM_in_latent)
                       "cem"           - standard TD-MPC CEM planning (agent.plan)
                       "deterministic" - raw policy network, no planning: pi(z, std=0)
        sample_final_action: Only used when policy == "cem_latent". If True, sample the
                     final latent action from the CEM search distribution instead of using
                     its mean. Passed through to CEM_in_latent.

    Returns:
        eval_metrics (dict): Evaluation statistics and episode rewards.
    """
    assert video_mode in {"first", "best_worst", "none"}, \
        "video_mode must be one of: 'first', 'best_worst', 'none'"
    assert policy in {"cem_latent", "cem", "deterministic"}, \
        "policy must be one of: 'cem_latent', 'cem', 'deterministic'"

    episode_rewards = []
    episode_frames = [] if video_mode == "best_worst" else None

    video_dir = None
    if save_dir and video_mode != "none":
        video_dir = os.path.join(save_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_in_ep = 0
        total_compute_time = 0

        record = (video_mode == "first" and ep == 0) or (video_mode == "best_worst")
        frames = [] if record else None

        episode_start = time.time()
        while not done:
            with torch.no_grad():
                compute_time_start = time.time()
                t0 = (step_in_ep == 0)
                if policy == "cem_latent":
                    action, *_ = agent.CEM_in_latent(obs, step=step, sample_final_action=sample_final_action, t0=t0)
                elif policy == "cem":
                    action = agent.plan(obs, eval_mode=True, step=step, t0=t0)
                else:  # "deterministic"
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    z = agent.model.h(obs_t)
                    action = agent.model.pi(z, std=0).squeeze(0)
                compute_time_end = time.time()
            obs, reward, done, _ = env.step(action.cpu().numpy())
            total_reward += reward
            step_in_ep += 1
            total_compute_time += (compute_time_end-compute_time_start)
            if record:
                try:
                    frame = env.render(mode='rgb_array', height=480, width=640, camera_id=0)
                except TypeError:
                    frame = env.render(mode='rgb_array')
                frames.append(frame)
        episode_end = time.time()

        episode_rewards.append(total_reward)
        if video_mode == "best_worst":
            episode_frames.append(frames)

        print(f"Episode {ep+1}/{n_episodes}: Reward = {total_reward:.3f}")

        if video_mode == "first" and ep == 0 and video_dir:
            video_path = os.path.join(video_dir, f"eval_step{step}_ep{ep+1:03d}.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"🎥 Saved first episode video: {video_path}")

    if video_mode == "best_worst" and video_dir and n_episodes > 0:
        best_idx = int(np.argmax(episode_rewards))
        worst_idx = int(np.argmin(episode_rewards))

        best_path = os.path.join(video_dir, f"eval_step{step}_best_ep{best_idx+1:03d}.mp4")
        imageio.mimsave(best_path, episode_frames[best_idx], fps=30)
        print(f"🏆 Saved best episode video: {best_path}")

        worst_path = os.path.join(video_dir, f"eval_step{step}_worst_ep{worst_idx+1:03d}.mp4")
        imageio.mimsave(worst_path, episode_frames[worst_idx], fps=30)
        print(f"💀 Saved worst episode video: {worst_path}")

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))

    eval_metrics = {
        "step": int(step),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_compute_duration": total_compute_time/1000,
        "episode_duration": episode_end-episode_start,
        "rewards": [] 
    }
    eval_metrics["rewards"] = [float(r) for r in episode_rewards]

    print(f"\nEvaluation Summary — Step {step}")
    print("-" * 25)
    print(f"Mean Reward: {mean_reward:.3f}")
    print(f"Std Reward:  {std_reward:.3f}")

    return eval_metrics
