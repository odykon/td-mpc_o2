#!/bin/bash
# run_schedule_ablation.sh
# Compares std/horizon schedule lengths: 10k vs 25k MuJoCo steps.
# Same task and seed for both runs.
# Run from the repo root: bash scripts/run_schedule_ablation.sh

set -e

TASK="walker-walk"
SEED=11

echo "=========================================="
echo "Schedule ablation  |  task=$TASK  seed=$SEED"
echo "=========================================="

echo ""
echo "--- Run 1/2: schedule_steps=10000 ---"
python3 scripts/train_o2_phased.py \
    cfg=cfgs/exp_phased.yaml \
    task="$TASK" \
    seed="$SEED" \
    mujoco_std_schedule_steps=10000 \
    mujoco_horizon_schedule_steps=10000 \
    exp_name=o2_phased_sched10k

echo ""
echo "--- Run 2/2: schedule_steps=25000 ---"
python3 scripts/train_o2_phased.py \
    cfg=cfgs/exp_phased.yaml \
    task="$TASK" \
    seed="$SEED" \
    mujoco_std_schedule_steps=25000 \
    mujoco_horizon_schedule_steps=25000 \
    exp_name=o2_phased_sched25k

echo ""
echo "Schedule ablation complete."
