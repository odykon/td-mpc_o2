#!/bin/bash
# Experiment: o2_diversity_01
# Two variants × 4 environments × 1 seed:
#   - with latent state (u + z)
#   - without latent state (u only)

python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_03.yaml       task=cheetah-run     seed=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_03.yaml       task=cheetah-run     seed=2
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_03.yaml       task=walker-walk     seed=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_03.yaml       task=walker-walk     seed=2
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_03.yaml       task=fish-swim       seed=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_03.yaml       task=fish-swim       seed=2
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_03.yaml       task=quadruped-run   seed=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_03.yaml       task=quadruped-run   seed=2