# Repo: TD-MPC O2

Thesis project extending TD-MPC with a latent action space decoder (the "O2" extension).
The goal is to train agents on DMControl tasks using CEM planning in a learned latent action space.

## Repo Structure

```
scripts/               # Runnable training entry points
  train_tdmpc.py            # Standard TD-MPC training (base agent, CEM planning)
  train_o2_ddpg.py          # O2 training: CEM in latent space + DDPG decoder update
  train_o2_phased.py        # Phased O2 experiment: TOLD-only → decoder warm-up → latent CEM
  train_tdmpc_resume.py     # TDMPC baseline: loads intermediate checkpoint, continues standard CEM
  run_phased_experiments.sh # Batch runner: 5 envs × 3 seeds (O2 phased + TDMPC baseline)

cfgs/                  # Custom YAML configs for training runs
  train_tdmpc.yaml     # Config for train_tdmpc.py — edit this to set hyperparams
  exp_phased.yaml      # Config for train_o2_phased.py — phased experiment hyperparams

notebooks/             # Debug/interactive training notebooks
  train_o2_ddpg_debug.ipynb  # Interactive O2 DDPG training loop

o2/                    # O2 extension — latent action space decoder
  tdmpc_o2.py          # TDMPC_O2 class: TDMPC subclass with decoder + value network
  action_decoder.py    # Decoder MLP architecture and initialisation
  planning.py          # DCEMethod (differentiable CEM), DCEMethod_v2 (with grad tracking), CEM_in_latent
                       #   → DCEMethod: raw value passed to LML (no normalisation)
                       #   → DCEMethod_v2: value standardised with detached mean+std before LML
  decoder_updates.py   # All decoder update strategies (DDPG, PG, PPO)
                       #   → action_decoder_DDPG_update_v2 returns: decoder_loss, decoder_grad_norm,
                       #     value_mean, jacobian_penalty, saturation, z_norm, u_norm
  training_utils.py    # Shared loop utilities: set_seed, update_tdmpc, update_decoder, update_decoder_pg
  logger.py            # CSVLogger — shared by all training scripts
  episode.py           # PGEpisode — extends Episode with on-policy fields
  eval_utils.py        # Evaluation, metrics, video saving (pandas import is lazy inside save_results)

tdmpc/                 # Original TD-MPC implementation (Hansen et al., 2022) — do not modify
  src/                 # Core source: TDMPC agent, TOLD model, ReplayBuffer, env wrappers, cfg parsing
  cfgs/                # YAML configs: default.yaml, tasks/<domain>.yaml

lml.py                 # LML soft top-k projection (Amos et al.) — external, do not modify

logs/                  # Training outputs (gitignored)
results/               # Evaluation outputs (gitignored)
```

## Key Design Concepts

**Standard TD-MPC** (`scripts/train_tdmpc.py`):
- Uses `TDMPC` agent from `tdmpc/src/algorithm/tdmpc.py`
- Plans with standard CEM via `agent.plan()`
- Updates world model via `agent.update()` (wrapped in `update_tdmpc` from `training_utils`)
- Logs to `logs/<task>/<modality>/<exp_name>/<seed>/train.csv` and `eval.csv`

**O2 Extension** (`scripts/train_o2_ddpg.py`):
- Uses `TDMPC_O2` agent from `o2/tdmpc_o2.py`
- Two-phase training:
  - Phase 1 (`step < decoder_start_steps`): standard CEM planning, TOLD updates only
  - Phase 2 (`step >= decoder_start_steps`): CEM in latent space, TOLD + decoder DDPG updates
- Decoder maps latent actions `u` → action sequences, conditioned on latent state `z`
- `DCEMethod` (differentiable CEM) is used for decoder training; `CEM_in_latent` (hard top-k) for rollouts

**Phased O2 Training** (`scripts/train_o2_phased.py` + `cfgs/exp_phased.yaml`):
- Single continuous loop, one buffer, three phases determined by MuJoCo step thresholds:
  - `tdmpc`  (0 → `mujoco_decoder_start_steps`): standard CEM planning, TOLD updates only (500/ep)
  - `warmup` (`mujoco_decoder_start_steps` → `mujoco_latent_start_steps`): standard CEM + TOLD (500/ep) + decoder (100/ep)
  - `o2`     (`mujoco_latent_start_steps` → `mujoco_train_steps`): latent CEM + TOLD (500/ep) + decoder (100/ep)
- Default thresholds: seed=4k, decoder warm-up=15k, latent CEM=20k, total=40k MuJoCo steps
- All thresholds specified in **raw MuJoCo interactions** and divided by `action_repeat` in `load_cfg`
- At the first step of the `o2` phase, saves both model and buffer as W&B artifacts (intermediate checkpoint)
- Final evaluation with video at end; nothing is saved locally — everything goes to W&B
- `PHASED_DEFAULTS` in the script holds all hyperparams; `exp_phased.yaml` mirrors them for override

**TDMPC Baseline** (`scripts/train_tdmpc_resume.py`):
- Downloads the intermediate model + buffer artifacts from W&B (saved by `train_o2_phased.py`)
- Continues standard CEM training for `mujoco_resume_steps` (default 20k) more MuJoCo interactions
- `step` starts from `step_offset` (= `mujoco_step_offset // action_repeat`) so W&B curves overlay directly with the O2 run
- Uses plain `TDMPC` agent; loads checkpoint with `strict=False` (tolerates decoder keys in TDMPC_O2 checkpoint)
- Artifact names: `intermediate_{task_safe}_seed{seed}:latest` and `intermediate_buffer_{task_safe}_seed{seed}:latest`

**W&B Conventions**:
- Project: `TDMPC_O2`, entity: `odysseaskon-national-technical-university-of-athens`
- `group=cfg.task` — separates environments in the UI; seeds within a group can be averaged/compared
- One W&B run per (task, seed); run name = `{exp_name}__seed{seed}`
- Nothing is saved locally — all artifacts (models, buffers, videos) go via `tempfile` → W&B → `os.unlink`
- Models logged as W&B Artifacts (type=`model`); buffers as type=`buffer`; video via `wandb.Video`

**O2 Defaults** — each O2 script defines its own `O2_DEFAULTS` dict, merged into the config before CLI args:
```python
O2_DEFAULTS = {
    'latent_action_dim':    128,
    'decoder_init':         True,
    'use_latent_state':     True,
    'dcem_batch_size':      64,
    'decoder_updates':      50,
    'told_updates':         500,
    'decoder_start_steps':  5000,
    'exp_name':             'o2_ddpg',
    # CEM hyperparams for latent-space planning (independent of standard CEM)
    'latent_num_samples':   32,
    'latent_num_elites':    8,
}
```
These can be overridden by a custom YAML or CLI args.

`episode_length` and `train_steps` are recomputed in `load_cfg` after all merges to ensure consistency when `action_repeat` is overridden (e.g. `action_repeat=1`).

**Loading pre-trained weights** — pass `load_model` and/or `load_buffer` in a YAML or as CLI args:
```yaml
load_model: /path/to/model.pt   # supports both agent.save() format and raw state_dict
load_buffer: /path/to/replay_buffer.pth
```
Works with checkpoints saved by the current script (`agent.save()`) and old-format checkpoints (`torch.save(agent.model.state_dict(), ...)`). The final model is always saved to `logs/.../final_model.pt` at the end of training.

## Config System

Configs are loaded by `parse_cfg` from `tdmpc/src/cfg.py`. Priority order (lowest to highest):
1. `tdmpc/cfgs/default.yaml` — base defaults
2. `tdmpc/cfgs/tasks/<domain>.yaml` — task-specific (mainly `action_repeat`)
3. `O2_DEFAULTS` (O2 scripts only)
4. Custom YAML passed as `cfg=my_cfg.yaml`
5. CLI args e.g. `task=walker-walk seed=1`

Use `make_cfg` from a training script to build a config programmatically in notebooks:
```python
import sys
sys.path.insert(0, '/kaggle/working/<repo-name>')
sys.path.insert(0, '/kaggle/working/<repo-name>/scripts')
from train_tdmpc import make_cfg
from omegaconf import OmegaConf
cfg = make_cfg('walker-walk', seed=1)
cfg.lr = 3e-4
OmegaConf.save(cfg, 'my_cfg.yaml')
# then: !python scripts/train_tdmpc.py cfg=my_cfg.yaml
```

Disable evaluations by setting `eval_episodes=0` in the config.

## Config System — Custom YAML fields

The custom YAML (e.g. `cfgs/train_tdmpc.yaml`) supports a few extra fields beyond the base defaults:

- `mujoco_episode_steps` / `mujoco_train_steps` — raw MuJoCo interaction counts; `episode_length` and `train_steps` are derived by dividing by `action_repeat` via `load_cfg`'s arithmetic evaluation
- `seed_steps_real` — pre-action-repeat seed steps; set `seed_steps: ${seed_steps_real}/${action_repeat}`
- Schedules (`std_schedule`, `horizon_schedule`) do **not** support interpolation inside the `linear(...)` string — hardcode the divided value directly

Run with:
```bash
python3 scripts/train_tdmpc.py cfg=cfgs/train_tdmpc.yaml
python3 scripts/train_tdmpc.py cfg=cfgs/train_tdmpc.yaml seed=2 task=cheetah-run
```

## Kaggle Usage

```bash
!git clone <repo-url>
%cd <repo-name>
!pip install dm-control omegaconf
!python scripts/train_tdmpc.py task=walker-walk seed=1
# or:
!python scripts/train_o2_ddpg.py task=walker-walk seed=1
```

GPU runtime must be enabled. `MUJOCO_GL=egl` is set automatically by the scripts.

Both scripts add `REPO_ROOT` and `REPO_ROOT/tdmpc/src` to `sys.path` at startup, so all imports
(`o2`, `algorithm`, `cfg`, `env`) resolve correctly without manual path setup.

## VS Code Remote SSH (kalymnos)

Connect to kalymnos directly from VS Code for a local-like editing experience.

**One-time setup (local machine):**
1. Install the **Remote - SSH** extension in VS Code
2. Add to `~/.ssh/config`:
   ```
   Host kalymnos
       HostName n12mavra.cs.ntua.gr
       Port 2222
       User okonias
   ```
3. Install the **Claude** extension in VS Code, then connect to kalymnos and install it on the remote too (`Install in SSH: kalymnos`)

**Connect:** `Cmd+Shift+P` → `Remote-SSH: Connect to Host` → `kalymnos`

VS Code installs the server on the remote automatically on first connect. The green `SSH: kalymnos` badge in the bottom-left confirms the connection.

**Note:** VS Code Server installs to `~/.vscode-server/` by default. Since `/` is nearly full, add this to your local VS Code settings (`settings.json`):
```json
"remote.SSH.serverInstallPath": {
    "kalymnos": "/gpu-data3/okonias/.vscode-server"
}
```

## Remote GPU Cluster Setup (kalymnos)

The cluster has no sudo access and the system disk (`/`) is nearly full. All large files must go on `/gpu-data3`.

**One-time environment setup:**
```bash
# Install miniconda to gpu-data3
cd /gpu-data3
curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /gpu-data3/okonias/miniconda3
source /gpu-data3/okonias/miniconda3/bin/activate
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -p /gpu-data3/okonias/envs/tdmpc_o2 python=3.9 -y
conda activate /gpu-data3/okonias/envs/tdmpc_o2

# Install packages — use TMPDIR and cache-dir to avoid writing to full system disk
mkdir -p /gpu-data3/okonias/tmp
TMPDIR=/gpu-data3/okonias/tmp pip install torch omegaconf --cache-dir /gpu-data3/okonias/pip_cache
TMPDIR=/gpu-data3/okonias/tmp pip install mujoco dm-control gym --only-binary :all: --cache-dir /gpu-data3/okonias/pip_cache
```

**Each session:**
```bash
source /gpu-data3/okonias/miniconda3/bin/activate
conda activate /gpu-data3/okonias/envs/tdmpc_o2
cd ~/projects/td-mpc_o2
# Force pull latest code from remote:
git fetch origin && git reset --hard origin/main
# Run experiments inside tmux so SSH disconnect doesn't kill training:
tmux new -s train        # or: tmux attach -t train
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=walker-walk seed=1
# Detach from tmux: Ctrl-B D   |   Reattach: tmux attach -t train
```

Note: `MUJOCO_GL=egl` is set automatically by the training scripts.

**GPU utilisation**: Small model + small batch → ~20% GPU usage. To increase:
- Set `batch_size: 2048` and `dcem_batch_size: 256` in the config (or via CLI) for higher throughput.

## Conventions

- Training scripts live in `scripts/` — one file per training procedure
- Shared utilities (update loops, `set_seed`) live in `o2/training_utils.py`
- `CSVLogger` lives in `o2/logger.py` — import it from there in all training scripts
- Decoder update functions are methods bound to `TDMPC_O2` — they take `self` as first arg
- `update_tdmpc` in `training_utils.py` works for both `TDMPC` and `TDMPC_O2` (uses `hasattr` guards)
- `tdmpc/` is upstream code — never modify it
- `lml.py` is external (Amos et al.) — never modify it
- `logs/` and `results/` are gitignored — never commit training outputs

## Running the Phased Experiments

Full experiment suite (5 envs × 3 seeds, O2 phased + TDMPC baseline):
```bash
bash scripts/run_phased_experiments.sh
```

Single run for testing:
```bash
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=walker-walk seed=1
```

TDMPC baseline for one run (requires intermediate artifacts already uploaded):
```bash
python3 scripts/train_tdmpc_resume.py task=walker-walk seed=1
```

## Planned Scripts

Future training scripts to be added to `scripts/`:
- `train_o2_pg.py`   — decoder trained with on-policy PG + value baseline
- `train_o2_ppo.py`  — decoder trained with PPO-KL

All will import shared utilities from `o2/training_utils.py`.
