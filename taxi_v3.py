import gym
from ray.rllib.algorithms.ppo import PPO 

# color print
from colorama import Fore, Back, Style


# using a predefined OpenAI Gym environment: Taxi-v3
# this environment is implemented with its custom init, reset, step, train, evaluate functions, and many more
algo = PPO(
	config = {
		"env": "Taxi-v3",
		"num_workers": 2,
		"evaluation_num_workers": 1,
		"evaluation_config": {
			"render_env": True,
		},
	}
)


for _ in range(30):
	print(algo.train())

algo.evaluate()
