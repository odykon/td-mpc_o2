#!/bin/bash
# Action repeat 1-4: decoder runs in sequence on GPU 0, TD-MPC-only baselines in sequence on GPU 3.
# Logs: runs/o2_base_ar<AR>_s1[_tdmpc].log   Results: runs/o2_base/cheetah-run_ar<AR>_s1[_tdmpc]/
#
# Usage (from repo root):  nohup bash o2_base/sweep.sh > runs/o2_base_sweep.log 2>&1 &

cd "$(dirname "$0")/.."
SEED=1

(
for AR in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 python -u o2_base/train.py --action_repeat $AR --seed $SEED \
        > runs/o2_base_ar${AR}_s${SEED}.log 2>&1
done
) &

(
for AR in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=3 python -u o2_base/train.py --action_repeat $AR --seed $SEED --tdmpc_only \
        > runs/o2_base_ar${AR}_s${SEED}_tdmpc.log 2>&1
done
) &

wait
