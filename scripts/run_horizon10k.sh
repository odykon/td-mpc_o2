#!/bin/bash
# run_horizon10k.sh
# Runs train_o2.py on cheetah-run, walker-walk, fish-swim, quadruped-walk
# (seeds 1 and 2) with horizon_schedule and std_schedule set to anneal over
# 10000 agent-steps instead of the default 25000.
# Run from the repo root: bash scripts/run_horizon10k.sh
#
# train_steps convention matches tdmpc's own default.yaml / tdmpc/src/train.py:
# train_steps is raw environment steps / action_repeat (so the env-step budget
# is comparable across tasks with different action_repeat), computed here as a
# literal integer per task rather than passed as "80000/${action_repeat}" on
# the CLI — train_o2.py's load_cfg() calls the same tdmpc/src/cfg.py::parse_cfg()
# as the vendored train.py, which eagerly resolves every raw CLI value before
# the task yaml (which defines action_repeat) is merged in, so that
# interpolation isn't available yet and raises InterpolationKeyError if passed
# on the CLI. seed_steps, unlike train_steps, is NOT divided by action_repeat
# here — that matches tdmpc/cfgs/default.yaml, which likewise leaves
# seed_steps a plain literal (5000) instead of an action_repeat expression.

set -e

TASKS=("cheetah-run" "walker-walk" "fish-swim" "quadruped-walk")
SEEDS=(1 2)
EXP_NAME=o2_horizon10k
ENV_STEPS=80000
SEED_STEPS=2000

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
            train_steps="$TRAIN_STEPS" \
            seed_steps="$SEED_STEPS" \
            horizon_schedule='linear(1,5,10000)' \
            std_schedule='linear(0.5,0.05,10000)'
    done
done

echo ""
echo "All $TOTAL runs complete."
