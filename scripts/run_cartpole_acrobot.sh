#!/bin/bash

python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=cheetah-run seed=5 use_wandb=True
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=cheetah-run seed=6 use_wandb=True

python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=walker-walk seed=5 use_wandb=True
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=walker-walk seed=6 use_wandb=True
