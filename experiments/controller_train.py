import itr_dvfs_controller_0 as ec

# color print
from colorama import Fore, Back, Style

import sys
import pandas as pd

from ray.rllib.algorithms.ppo import PPO

debug = True

featurized_logs_file = sys.argv[1]
featurized_logs_test_file = sys.argv[2]
df = pd.read_csv(featurized_logs_file, sep = ',')
df_test = pd.read_csv(featurized_logs_test_file, sep = ',')

if debug:
	print()
	print()
	print(Fore.BLACK + Back.GREEN + "df: " + Style.RESET_ALL)
	print(df)
	print()
	print()
	print(Fore.BLACK + Back.GREEN + "df_test: " + Style.RESET_ALL)
	print(df_test)
	print()
	print()


algo = PPO(
	config = {
		"env": ec.EnergyCorridor,
		"env_config": {
			"df": df,
			"reset_count": 0,
			"success_count": 0,
			"step_count": 0,
		},
		"framework": 'torch',
		"num_workers": 1,
		"horizon": 20,
		"seed": 4,
		"gamma": 0.5,
#		"lr": 1e-6,
	}
)

# test seed
# [np.random.random() for i in range(10)]
def fix_seeds(seed=0):
	pass
#	np.random.seed(seed)
#	torch.manual_seed(seed)
#	torch.use_deterministic_algorithms(True)

for i in range(1):
	results = algo.train()
	print(Fore.BLACK + Back.BLUE + f"Iter: {i}, episode_reward_mean = {results['episode_reward_mean']}" + Style.RESET_ALL)

#checkpoint_file = algo.save("trained_models/RL_model_0_gamma_9")


