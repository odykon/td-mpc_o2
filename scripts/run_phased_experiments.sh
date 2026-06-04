#!/bin/bash
# run_phased_experiments.sh
# Runs all experiments sequentially (5 envs × 2 seeds).

set +e

ENVS=("cheetah-run" "walker-run" "fish-swim" "cup-catch" "acrobot-swingup")
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
        python3 scripts/train_o2_phased.py \
            cfg=cfgs/exp_phased.yaml \
            task="$TASK" \
            seed="$SEED"
    done
done

echo ""
echo "All $TOTAL runs complete."
