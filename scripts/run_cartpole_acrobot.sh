#!/bin/bash

python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=cartpole-swingup seed=1 use_wandb=True
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=cartpole-swingup seed=2 use_wandb=True
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=acrobot-swingup seed=1 use_wandb=True
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=acrobot-swingup seed=2 use_wandb=True

python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=cartpole-swingup seed=1 use_wandb=True exp_name=TDMPC_normal mujoco_decoder_start_steps=80008 mujoco_latent_start_steps=80016
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=cartpole-swingup seed=2 use_wandb=True exp_name=TDMPC_normal mujoco_decoder_start_steps=80008 mujoco_latent_start_steps=80016
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=acrobot-swingup seed=1 use_wandb=True exp_name=TDMPC_normal mujoco_decoder_start_steps=80008 mujoco_latent_start_steps=80016
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=acrobot-swingup seed=2 use_wandb=True exp_name=TDMPC_normal mujoco_decoder_start_steps=80008 mujoco_latent_start_steps=80016
