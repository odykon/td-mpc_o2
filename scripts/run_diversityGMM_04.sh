python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=quadruped-run seed=4 gmm_init='kmeans++' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=quadruped-run seed=5 gmm_init='kmeans++' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=quadruped-run seed=4 gmm_init='forgy' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=quadruped-run seed=5 gmm_init='forgy' kmeans_iters=1

python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=4 gmm_init='kmeans++' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=5 gmm_init='kmeans++' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=6 gmm_init='forgy' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=7 gmm_init='forgy' kmeans_iters=1

python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=fish-swim seed=4 gmm_init='kmeans++' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=fish-swim seed=5 gmm_init='kmeans++' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=fish-swim seed=6 gmm_init='forgy' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=fish-swim seed=7 gmm_init='forgy' kmeans_iters=1


