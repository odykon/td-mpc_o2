#!/bin/bash
# TD-MPC cheetah-run, 50 episodes, AR 3/4 x seeds 1/2.
# AR 1/2 were trained on Kaggle and imported into the same experiment (scripts/import_legacy_run.py).
# episode_length / seed_steps are passed as ints: the yaml defaults give floats for AR=3.
# train_steps = (EPISODES - 1) * EP_LEN because the training loop includes train_steps.
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
EXP=2026-11-03_50ep-ar-sweep
EPISODES=50; SEED_EPISODES=4
for AR in 3 4; do
  EP_LEN=$(( 1000 / AR ))
  for SEED in 1 2; do
    python3 scripts/train_tdmpc.py cfg=cfgs/train_tdmpc.yaml task=cheetah-run \
      seed=$SEED action_repeat=$AR episode_length=$EP_LEN \
      train_steps=$(( (EPISODES - 1) * EP_LEN )) seed_steps=$(( SEED_EPISODES * EP_LEN )) \
      exp=$EXP variant=ar$AR save_model=true
  done
done
