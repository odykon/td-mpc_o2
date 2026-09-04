#!/bin/bash
# run_o2_200k.sh
# TDMPC_O2 (latent-action CEM) runs using scripts/train_o2.py, mirroring the
# structure of run_std_tdmpc_300k.sh.
#
# Environments: cheetah-run, walker-walk, fish-swim, quadruped-walk
# Seeds:        1, 2, 3
# Planning:     latent_num_samples=16, latent_num_elites=4
# Length:       200000 environment steps per task (train_steps = 200000/action_repeat,
#               same convention as tdmpc/cfgs/default.yaml's train_steps=500000/action_repeat,
#               so runs stay comparable across tasks with different action_repeat).
#               train_steps is computed here as a literal integer (not passed as
#               "200000/${action_repeat}") because train_o2.py's load_cfg() calls
#               the same tdmpc/src/cfg.py::parse_cfg() as the vendored train.py,
#               which eagerly resolves every raw CLI value before merging in the
#               task yaml that defines action_repeat, so that interpolation isn't
#               available yet and raises InterpolationKeyError if passed on the CLI.
#
# Run from the repo root: bash scripts/run_o2_200k.sh
# wandb: use_wandb/wandb_project/wandb_entity default to on in train_o2.py's own
# DEFAULTS (cfgs/o2_default.yaml sourced), so no need to pass them here.

set -e

export CUDA_VISIBLE_DEVICES=1

TASKS=("cheetah-run" "walker-walk" "fish-swim" "quadruped-walk")
SEEDS=(1 2 3)
EXP_NAME=o2_200k
ENV_STEPS=200000
LATENT_NUM_SAMPLES=16
LATENT_NUM_ELITES=4
SEED_STEPS=5000

# action_repeat per task, from tdmpc/cfgs/tasks/<domain>.yaml (falls back to
# tasks/default.yaml's action_repeat=4 for any domain without its own file).
declare -A ACTION_REPEAT=(
    ["cheetah-run"]=4
    ["walker-walk"]=2
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
        python3 scripts/train_o2.py \
            task="$TASK" \
            seed="$SEED" \
            exp_name="$EXP_NAME" \
            latent_num_samples="$LATENT_NUM_SAMPLES" \
            latent_num_elites="$LATENT_NUM_ELITES" \
            seed_steps="$SEED_STEPS" \
            train_steps="$TRAIN_STEPS"
    done
done

echo ""
echo "All $TOTAL runs complete."
