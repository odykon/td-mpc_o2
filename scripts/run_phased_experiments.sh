#!/bin/bash
# run_phased_experiments.sh
# For each (task, seed): runs the O2 phased training, then immediately runs
# the TDMPC continue script which loads the Phase 1 checkpoint saved by the
# phased run. Both scripts finish before moving to the next (task, seed).

set +e

ENVS=("cheetah-run" "walker-run" "fish-swim" "cup-catch" "acrobot-swingup")
SEEDS=(1 2 3)

TOTAL=$(( ${#ENVS[@]} * ${#SEEDS[@]} ))
COUNT=0

for TASK in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COUNT=$(( COUNT + 1 ))
        echo ""
        echo "=========================================="
        echo "Pair $COUNT / $TOTAL  |  task=$TASK  seed=$SEED"
        echo "=========================================="

        echo "--- Phase: O2 phased ---"
        python3 scripts/train_o2_phased.py \
            cfg=cfgs/exp_phased.yaml \
            task="$TASK" \
            seed="$SEED"

        echo "--- Phase: TDMPC continue ---"
        python3 scripts/train_tdmpc_continue.py \
            cfg=cfgs/exp_phased.yaml \
            task="$TASK" \
            seed="$SEED"
    done
done

echo ""
echo "All $TOTAL pairs complete."
