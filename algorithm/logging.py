import pandas as pd
import os
import torch
from datetime import datetime
from zoneinfo import ZoneInfo
from omegaconf import OmegaConf
import json
import requests

def save_results(cfg, all_metrics, base_dir="results", timezone="Europe/Athens"):
    # Create readable date-time string (no seconds)
    local_time = datetime.now(ZoneInfo(timezone))
    timestamp = local_time.strftime("%Y-%m-%d_%Hh%M")

    exp_name = getattr(cfg, "exp_name", "experiment")
    save_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Convert config and save
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    pd.DataFrame(list(cfg_dict.items()), columns=["key", "value"]).to_csv(
        os.path.join(save_dir, "config.csv"), index=False
    )

    # Save metrics
    pd.DataFrame(all_metrics).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

    print(f"\n✅ Saved training results (Athens time) to:\n  {save_dir}")
    print("  ├── config.csv")
    print("  └── metrics.csv")

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