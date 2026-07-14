python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=fish-swim seed=2 gmm_init='kmeans++' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=fish-swim seed=3 gmm_init='kmeans++' kmeans_iters=1

python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=2 gmm_init='kmeans++' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=3 gmm_init='kmeans++' kmeans_iters=1

python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=walker-walk seed=2 gmm_init='kmeans++' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=walker-walk seed=3 gmm_init='kmeans++' kmeans_iters=1

python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=quadruped-run seed=2 gmm_init='kmeans++' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=quadruped-run seed=3 gmm_init='kmeans++' kmeans_iters=1
