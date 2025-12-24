import pandas as pd
import os
import torch
from datetime import datetime
from zoneinfo import ZoneInfo
from omegaconf import OmegaConf
import json
import requests

import numpy as np
import torch
import imageio
import os

def make_save_dir_path(cfg, base_dir="results", timezone="Europe/Athens"):
    local_time = datetime.now(ZoneInfo(timezone))
    timestamp = local_time.strftime("%Y-%m-%d_%Hh%M")
    exp_name = getattr(cfg, "exp_name", "experiment")
    save_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
    return save_dir

def evaluate_agent(env, agent, cfg, step, cem=False, n_episodes=5, save_dir=None, video_mode="none"):
    """
    Evaluate the agent and optionally save videos.

    Args:
        env: Environment (DMControl or Gym-like)
        agent: Agent with DCEMethod(obs, step, t0)
        cfg: Config (optional)
        step: Current training step number
        n_episodes: Number of evaluation episodes
        save_dir: Directory where videos are saved (optional)
        video_mode: "first", "best_worst", or "none"

    Returns:
        eval_metrics (dict): Evaluation statistics and episode rewards.
    """
    assert video_mode in {"first", "best_worst", "none"}, \
        "video_mode must be one of: 'first', 'best_worst', 'none'"

    episode_rewards = []
    episode_frames = [] if video_mode == "best_worst" else None

    # Create video folder if needed
    video_dir = None
    if save_dir and video_mode != "none":
        video_dir = os.path.join(save_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_in_ep = 0

        # Decide if this episode should record frames
        record = (video_mode == "first" and ep == 0) or (video_mode == "best_worst")
        frames = [] if record else None

        while not done:
            with torch.no_grad():
                if cem: 
                    action = agent.plan(obs, eval_mode =True, step=step_in_ep, t0=(step_in_ep == 0))
                else:    
                    action, _, _, _, _ = agent.DCEMethod(obs, step=step_in_ep, t0=(step_in_ep == 0))
            obs, reward, done, _ = env.step(action.cpu().numpy())
            total_reward += reward
            step_in_ep += 1

            if record:
                try:
                    frame = env.render(mode='rgb_array', height=480, width=640, camera_id=0)
                except TypeError:
                    frame = env.render(mode='rgb_array')
                frames.append(frame)

        episode_rewards.append(total_reward)
        if video_mode == "best_worst":
            episode_frames.append(frames)

        print(f"Episode {ep+1}/{n_episodes}: Reward = {total_reward:.3f}")

        # Save video if mode="first"
        if video_mode == "first" and ep == 0 and video_dir:
            video_path = os.path.join(video_dir, f"eval_step{step}_ep{ep+1:03d}.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"🎥 Saved first episode video: {video_path}")

    # --- Handle best/worst video saving ---
    if video_mode == "best_worst" and video_dir and n_episodes > 0:
        best_idx = int(np.argmax(episode_rewards))
        worst_idx = int(np.argmin(episode_rewards))

        # Save best
        best_path = os.path.join(video_dir, f"eval_step{step}_best_ep{best_idx+1:03d}.mp4")
        imageio.mimsave(best_path, episode_frames[best_idx], fps=30)
        print(f"🏆 Saved best episode video: {best_path}")

        # Save worst
        worst_path = os.path.join(video_dir, f"eval_step{step}_worst_ep{worst_idx+1:03d}.mp4")
        imageio.mimsave(worst_path, episode_frames[worst_idx], fps=30)
        print(f"💀 Saved worst episode video: {worst_path}")

    # --- Compute statistics ---
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))

    eval_metrics = {
        "step": int(step),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }
    for i, r in enumerate(episode_rewards, start=1):
        eval_metrics[f"ep{i}_reward"] = float(r)

    print(f"\nEvaluation Summary — Step {step}")
    print("-" * 25)
    print(f"Mean Reward: {mean_reward:.3f}")
    print(f"Std Reward:  {std_reward:.3f}")

    return mean_metrics

def save_results(cfg, episode_metrics, save_dir, evaluation_metrics=None, step=None):
    os.makedirs(save_dir, exist_ok=True)

    # Save config once
    cfg_path = os.path.join(save_dir, "config.csv")
    if not os.path.exists(cfg_path):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        pd.DataFrame(list(cfg_dict.items()), columns=["key", "value"]).to_csv(cfg_path, index=False)

    # Combine episode + eval metrics
    all_metrics = episode_metrics.copy()

    if evaluation_metrics is not None:
        # Merge evaluation metrics into the same row
        all_metrics.update(evaluation_metrics)
    else:
        # If no evaluation, ensure eval_* keys exist (filled with NaN)
        all_metrics.update({
            "mean_reward": np.nan,
            "std_reward": np.nan
        })

    # Add step and timestamp
    if step is not None:
        all_metrics["step"] = step
    all_metrics["timestamp"] = datetime.now().isoformat(timespec="seconds")

    # Save to CSV (append or create)
    metrics_path = os.path.join(save_dir, "metrics.csv")
    df = pd.DataFrame([all_metrics])
    if os.path.exists(metrics_path):
        df.to_csv(metrics_path, mode="a", header=False, index=False)
    else:
        df.to_csv(metrics_path, index=False)

    print(f"✅ Results saved to: {save_dir}")
    return save_dir



def save_model_and_buffer(agent, buffer, save_dir, model_name="model", buffer_name="replay_buffer"):
    """
    Saves the agent model and replay buffer to the specified directory.

    Args:
        agent: your agent object (must have `model` attribute)
        buffer: replay buffer object (must be picklable with torch.save)
        save_dir: directory to save into (should already exist)
        model_name: base filename for the model
        buffer_name: base filename for the replay buffer
    """

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # --- Save model weights ---
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(agent.model.state_dict(), model_path)

    # --- Save replay buffer ---
    buffer_path = os.path.join(save_dir, f"{buffer_name}.pth")
    torch.save(buffer.__dict__, buffer_path)

    # --- Print confirmation ---
    model_size = os.path.getsize(model_path) / 1e6
    buffer_size = os.path.getsize(buffer_path) / 1e6
    print(f"\n💾 Saved model and buffer to {save_dir}")
    print(f"  ├── {model_name}.pth  ({model_size:.2f} MB)")
    print(f"  └── {buffer_name}.pth  ({buffer_size:.2f} MB)")

    return model_path, buffer_path

def save_notebook_as_py(output_path=''):
    """
    Save the current Google Colab notebook as a Python (.py) script.

    Parameters:
        output_path (str): The output file path for the .py script.
    """
    output_path = output_path+ '/notebook.py'

    try:
        # Get the notebook metadata (requires Colab environment)
        from google.colab import _message
        notebook_data = _message.blocking_request("get_ipynb")

        # Extract code cells
        cells = notebook_data['ipynb']['cells']
        code_cells = [
            "# %%\n" + "".join(cell['source'])
            for cell in cells if cell['cell_type'] == 'code'
        ]

        # Combine into one script
        script_content = "\n\n".join(code_cells)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated from Colab Notebook\n\n")
            f.write(script_content)

        print(f"✅ Notebook saved as {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"❌ Error saving notebook: {e}")
