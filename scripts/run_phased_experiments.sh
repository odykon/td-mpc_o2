#!/bin/bash
# run_phased_experiments.sh
# Runs all 15 phased training experiments (5 envs × 3 seeds) sequentially.
# Run from the repo root: bash scripts/run_phased_experiments.sh

set +e

ENVS=("cheetah-run" "walker-walk" "hopper-hop" "humanoid-walk" "cartpole-swingup")
SEEDS=(1 2 3)

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
echo "All $TOTAL O2 experiments complete."
echo ""
echo "Running TDMPC baseline..."
COUNT=0
for TASK in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COUNT=$(( COUNT + 1 ))
        echo ""
        echo "=========================================="
        echo "Baseline $COUNT / $TOTAL  |  task=$TASK  seed=$SEED"
        echo "=========================================="
        python3 scripts/train_tdmpc_resume.py \
            task="$TASK" \
            seed="$SEED"
    done
done

echo ""
echo "All experiments complete."
