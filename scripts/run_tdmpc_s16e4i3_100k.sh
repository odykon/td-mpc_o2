#!/bin/bash
# run_std_tdmpc_300k.sh
# Standard TD-MPC baseline runs using the original (vendored) tdmpc/src/train.py,
# unmodified, for maximum fidelity to the original implementation.
#
# Environments: cheetah-run, walker-walk, fish-swim, quadruped-walk
# Seeds:        1, 2
# Planning:     num_samples=16, num_elites=4, iterations=3
# Length:       100000 environment steps per task, i.e. train_steps = 100000/action_repeat,
#               same convention as tdmpc/cfgs/default.yaml's train_steps=500000/action_repeat,
#               so runs stay comparable across tasks with different action_repeat.
#               train_steps is computed here as a literal integer (not passed as
#               "100000/${action_repeat}") because tdmpc/src/cfg.py's parse_cfg()
#               eagerly resolves every raw CLI value before merging in the task yaml
#               that defines action_repeat, so that interpolation isn't available yet
#               and raises InterpolationKeyError if passed on the CLI.
#
# Run from the repo root: bash scripts/run_std_tdmpc_300k.sh
# Logs land in tdmpc/logs/<task>/state/<exp_name>/<seed>/ (original repo layout).

set -e

export CUDA_VISIBLE_DEVICES=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASKS=("cheetah-run" "fish-swim" "quadruped-walk")
SEEDS=(1 2)
EXP_NAME=tdmpc_s16e4i3_100k_w/policy_trajs
ENV_STEPS=100000
NUM_SAMPLES=16
NUM_ELITES=4
ITERS=3
MIXTURE_COEFF=0.1875 #corresponds to 2 pi trajs
SAVE_MODEL=True



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
        (
            cd "$REPO_ROOT/tdmpc"
            python src/train.py \
                task="$TASK" \
                seed="$SEED" \
                exp_name="$EXP_NAME" \
                num_samples="$NUM_SAMPLES" \
                num_elites="$NUM_ELITES" \
                iterations="$ITERS" \
                mixture_coeff="$MIXTURE_COEFF" \
                save_model="$SAVE_MODEL" \
                train_steps="$TRAIN_STEPS" \
                use_wandb=True \
                wandb_project='TDMPC_O2' \
                wandb_entity='odysseaskon-national-technical-university-of-athens'
        )
    done
done

echo ""
echo "All $TOTAL runs complete."