#!/bin/bash
# run_no_warmstart_experiments.sh
# Trains standard TD-MPC (no CEM mean warm-starting) on cheetah-run and
# walker-walk, seeds 1 and 2.

set +e

ENVS=("cheetah-run" "walker-walk")
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
        python3 scripts/train_tdmpc_no_warmstart.py \
            cfg=cfgs/train_tdmpc_no_warmstart.yaml \
            task="$TASK" \
            seed="$SEED"
    done
done

echo ""
echo "All $TOTAL runs complete."
