#!/bin/bash
# Decoder runs for action repeat 1-4, one after another, with scale monitoring.
# Results: runs/o2_base/cheetah-run_ar<AR>_s1_monitor/   Logs: runs/o2_base_ar<AR>_s1_monitor.log
#
# Usage (from repo root):  GPU=3 nohup bash o2_base/sweep_monitor.sh > runs/o2_base_monitor_sweep.log 2>&1 &

cd "$(dirname "$0")/.."
SEED=1
GPU=${GPU:-0}

for AR in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$GPU python -u o2_base/train.py --action_repeat $AR --seed $SEED --tag monitor \
        > runs/o2_base_ar${AR}_s${SEED}_monitor.log 2>&1
done
