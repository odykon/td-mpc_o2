#!/bin/bash
# run_immediate_resume_experiments.sh
# Runs all 10 o2_immediate_resume experiments (5 envs × 2 seeds) sequentially.
# Run from the repo root: bash scripts/run_immediate_resume_experiments.sh

set -e

ENVS=("cheetah-run" "walker-walk" "hopper-hop" "humanoid-walk" "cartpole-swingup")
SEEDS=(1 2)

TOTAL=$(( ${#ENVS[@]} * ${#SEEDS[@]} ))
COUNT=0

for TASK in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COUNT=$(( COUNT + 1 ))
        echo ""
        echo "=========================================="
        echo "Run $COUNT / $TOTAL  |  task=$TASK  seed=$SEED"
        echo "=========================================="
        python3 scripts/train_o2_immediate.py \
            cfg=cfgs/exp_immediate.yaml \
            task="$TASK" \
            seed="$SEED" \
            load_model="checkpoints/$TASK/$SEED/model.pt" \
            load_buffer="checkpoints/$TASK/$SEED/buffer.pth" \
            mujoco_step_offset=20000
    done
done

echo ""
echo "All $TOTAL experiments complete."
