#!/bin/bash

python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=cheetah-run seed=1 use_wandb=True
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=cheetah-run seed=2 use_wandb=True

python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=walker-walk seed=1 use_wandb=True
python3 scripts/train_o2_phased.py cfg=cfgs/exp_phased.yaml task=walker-walk seed=2 use_wandb=True
