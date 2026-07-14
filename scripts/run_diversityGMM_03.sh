python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=1 gmm_init='forgy' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=2 gmm_init='forgy' kmeans_iters=2
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=1 gmm_init='kmeans++' kmeans_iters=1
python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=2 gmm_init='kmeans++' kmeans_iters=2

#python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=1 gmm_init='forgy' gmm_num_samples=256
#python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=2 gmm_init='forgy' gmm_num_samples=256
#python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=1 gmm_init='kmeans++' gmm_num_samples=256 
#python3 scripts/train_o2_phased.py cfg=cfgs/exp_diversity_04.yaml task=cheetah-run seed=2 gmm_init='kmeans++' gmm_num_samples=256

