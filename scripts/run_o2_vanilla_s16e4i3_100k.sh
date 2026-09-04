#!/bin/bash
# run_o2_vanilla_300k.sh
# Same runs as run_std_tdmpc_300k.sh (same tasks, seeds, step budget, wandb
# project) but using scripts/train_o2_vanilla.py — the TD-MPC + O2 latent
# decoder variant kept as close as possible to the vendored tdmpc/src/train.py.
#
# Environments: cheetah-run, walker-walk, fish-swim, quadruped-walk
# Seeds:        1, 2
# Planning:     latent_num_samples=16, latent_num_elites=4, iterations=3, cem_warmstart=true
# Length:       100000 environment steps per task, i.e. train_steps = 100000/action_repeat,
#               same convention as tdmpc/cfgs/default.yaml's train_steps=500000/action_repeat,
#               so runs stay comparable across tasks with different action_repeat.
#               train_steps is computed here as a literal integer (not passed as
#               "100000/${action_repeat}") because train_o2_vanilla.py's load_cfg()
#               calls the same tdmpc/src/cfg.py::parse_cfg() as the vendored train.py,
#               which eagerly resolves every raw CLI value before merging in the task
#               yaml that defines action_repeat, so that interpolation isn't available
#               yet and raises InterpolationKeyError if passed on the CLI.
#
# Run from the repo root: bash scripts/run_o2_vanilla_300k.sh
# (train_o2_vanilla.py uses absolute paths, so no `cd` into tdmpc/ is needed —
# logs land in <repo-root>/logs/<task>/state/<exp_name>/<seed>/.)

set -e

export CUDA_VISIBLE_DEVICES=2

TASKS=("cheetah-run" "fish-swim")
SEEDS=(1 2)
EXP_NAME=o2_vanilla_s16e4i3_100k_w/policy_trajs
ENV_STEPS=100000
LATENT_NUM_SAMPLES=16
LATENT_NUM_ELITES=4
ITERATIONS=3
DCEM_ITERATIONS=3
CEM_WARMSTART=true
USE_LATENT_STATE=true
SAVE_MODEL=true
NUM_PI_TRAJS=2

# action_repeat per task, from tdmpc/cfgs/tasks/<domain>.yaml (falls back to
# tasks/default.yaml's action_repeat=4 for any domain without its own file).
declare -A ACTION_REPEAT=(
    ["cheetah-run"]=4
    #["walker-walk"]=2
    ["fish-swim"]=4
    ["quadruped-walk"]=4
)

TOTAL=$(( ${#TASKS[@]} * ${#SEEDS[@]} ))
COUNT=0

for TASK in "${TASKS[@]}"; do
    TRAIN_STEPS=$(( ENV_STEPS / ACTION_REPEAT[$TASK] ))
    for SEED in "${SEEDS[@]}"; do
        COUNT=$(( COUNT + 1 ))
        echo ""
        echo "=========================================="
        echo "Run $COUNT / $TOTAL  |  task=$TASK  seed=$SEED  train_steps=$TRAIN_STEPS"
        echo "=========================================="
        python scripts/train_o2_vanilla.py \
            task="$TASK" \
            seed="$SEED" \
            exp_name="$EXP_NAME" \
            latent_num_samples="$LATENT_NUM_SAMPLES" \
            latent_num_elites="$LATENT_NUM_ELITES" \
            iterations="$ITERATIONS" \
            dcem_iterations="$DCEM_ITERATIONS" \
            cem_warmstart="$CEM_WARMSTART" \
            use_latent_state="$USE_LATENT_STATE" \
            num_pi_trajs="$NUM_PI_TRAJS" \
            train_steps="$TRAIN_STEPS" \
            use_wandb=True \
            wandb_project='TDMPC_O2' \
            wandb_entity='odysseaskon-national-technical-university-of-athens' \
            wandb_tags='o2,tdmpc,latent' \
            save_model="$SAVE_MODEL"
    done
done

echo ""
echo "All $TOTAL runs complete."
