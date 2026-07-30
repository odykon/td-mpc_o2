python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=fish-swim seed=1 log_det_target=-2 future_sat_coeff=0 exp_name=fish_work
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=fish-swim seed=2 log_det_target=-2 future_sat_coeff=0.01 exp_name=fish_work

python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=fish-swim seed=3 log_det_target=-4 future_sat_coeff=0 exp_name=fish_work
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=fish-swim seed=4 log_det_target=-4 future_sat_coeff=0.01 exp_name=fish_work
