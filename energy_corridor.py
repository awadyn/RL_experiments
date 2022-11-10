# EXPERIMENT 1

# this test script will replicate the simple corridor RL example
# we will define an EnergyCorridor environment
# initialized as a Box of N rows and M columns, each column being a log metric, and each row being a full run
# resetting resets that same exact box (using the same samples)
# stepping simply takes a step left or right, searching for minimum energy


import gym
from ray.rllib.algorithms.ppo import PPO

import pandas as pd
import numpy as np
import utils

# color print
from colorama import Fore, Back, Style

debug = True
step_count = 0



def prepare_action_dicts(df):
	def get_knob_dict(knob):
		l = np.sort(df[knob].unique())
		l_p1 = np.roll(l, shift=-1)
		l_p1[-1] = -1 #invalid choice
		l_m1 = np.roll(l, shift=1)
		l_m1[0] = -1 #invalid choice
		d = {}
		for idx, elem in enumerate(l):
			d[elem] = {-1: l_m1[idx], 0: elem, 1: l_p1[idx]}
		return d
	d = {}
	knob_list = []
	for knob in ['itr', 'dvfs']:
		knob_list.append(knob)
		d[knob] = get_knob_dict(knob)	
	return d, knob_list



def init_dataset(df):
	reward_cols = ['joules_99', 'joules_per_interrupt', 'time_per_interrupt']

	df_state = df.set_index(['itr', 'dvfs', 'qps']).drop(reward_cols, axis=1)
	df_reward = df.set_index(['itr', 'dvfs', 'qps'])[reward_cols]
	state_dict = df_state.T.to_dict()
	for key in state_dict:
		state_dict[key] = np.array(list(state_dict[key].values()))
	reward_dict = df_reward.T.to_dict()
	action_dict, knob_list = prepare_action_dicts(df)
	key_list = list(state_dict.keys())

	if debug:
		print(Fore.BLACK + Back.GREEN + "state_dict: " + Style.RESET_ALL)
		print(df_state)
		print(Fore.BLACK + Back.GREEN + "reward_dict: " + Style.RESET_ALL)
		print(df_reward)
		print(Fore.BLACK + Back.GREEN + "action_dict: " + Style.RESET_ALL)
		print(action_dict)
		print(Fore.BLACK + Back.GREEN + "knob_list: " + Style.RESET_ALL)
		print(knob_list)
		print(Fore.BLACK + Back.GREEN + "key_list: " + Style.RESET_ALL)
		print(key_list)

	return state_dict, reward_dict, action_dict, knob_list, key_list



# problem definition: 
#	a corridor in which an agent must move right or left to find minimum energy
#	moves imply changing only DVFS
class EnergyCorridor(gym.Env):
	def __init__(self, config):
		df = config["df"]

		state_dict, reward_dict, action_dict, knob_list, key_list = init_dataset(df)
		# for the purpose of this experiment - changing only dvfs
		self.actions = action_dict['dvfs']

		# dictionaries of X log encodings with ITR-delay = 100 and QPS = 40000
		self.state_space = {}
		self.reward_space = {}
		self.key_space = []
		for key in list(state_dict.keys()):
			if key[2] == 40000:
				self.state_space[key] = state_dict[key]
				self.reward_space[key] = reward_dict[key]
				self.key_space.append(key)

		# |observation_space| = as many normalized feature vectors as is available
		N = self.state_space[self.key_space[0]].shape[0]
		self.observation_space = gym.spaces.Box(low = np.zeros((N)), high = np.inf*np.ones(N))

		# goal key = key that yields min energy
		joules_99 = list({k:v['joules_99'] for k,v in self.reward_space.items()}.values())
		min_joules_99 = min(joules_99)
		self.goal_key = self.key_space[joules_99.index(min_joules_99)]
		# default start key
		self.cur_key = self.key_space[0]

		# |action_space| = 3 (increase DVFS, decrease DVFS, or keep unchanged)
		self.action_space = gym.spaces.Discrete(3)

		if debug:
			print(Fore.BLACK + Back.RED + "state_space =  "  + Style.RESET_ALL)
			print(self.state_space)
			print(Fore.BLACK + Back.RED + "reward_space = " + Style.RESET_ALL)
			print(self.reward_space)
			print(Fore.BLACK + Back.RED + "key_space" + Style.RESET_ALL)
			print(self.key_space)
			print(Fore.BLACK + Back.RED + "action_space" + Style.RESET_ALL)
			print(self.action_space)
			print(Fore.BLACK + Back.RED + "observation_space" + Style.RESET_ALL)
			print(self.observation_space)
			print(Fore.BLACK + Back.RED + "goal_key, min_joules_99" + Style.RESET_ALL)
			print(self.goal_key, min_joules_99)



	# runs everytime 1 episode of many attempted moves is completed
	def reset(self):
		global step_count

		if debug:
			print(Fore.BLACK + Back.RED + "resetting env... step_count = " + str(step_count) + Style.RESET_ALL)
		step_count = 0

		# a simple default reset 
		self.cur_key = self.key_space[0]
		print("cur_key = ", self.cur_key)
		print("reset state = ", self.state_space[self.cur_key])
		return self.state_space[self.cur_key]


	def step(self, action):
		global step_count

		new_key = list(self.cur_key)
		new_dvfs = self.actions[self.cur_key[1]][action - 1] 
		new_key[1] = new_dvfs
		new_key = tuple(new_key)

		# reward = 1 when reaching end_pos and -0.1 at every step otherwise
		done = (new_key == self.goal_key)
		reward = 1.0 if done else -0.1

		if debug:
			print(Fore.BLACK + Back.GREEN + "STEP: action =  " + str(action - 1) + ", reward = " + str(reward) + ", done = " + str(done) + Style.RESET_ALL)
			print(Fore.BLACK + Back.GREEN + "new key: " + Style.RESET_ALL)
			print(new_key)
		step_count += 1

		new_state = self.state_space[new_key]

		return new_state, reward, done, {}



df = pd.read_csv('./features/logs_0_ebbrt_percentiles.csv', sep = ' ')
df = utils.normalize(df)


if debug:
	print()
	print()
	print('----------------------------------------------------------')
	print('FEATURES: ------------------------------------------------')
	print('----------------------------------------------------------')
	print(Fore.BLACK + Back.GREEN + "df: " + Style.RESET_ALL)
	print(df)


# defining the training algorithm
# PPO is the mathematical model
algo = PPO(
	config = {
		"env": EnergyCorridor,
		"env_config": {
			"df": df,
		},
		"num_workers": 1,
	}
)

for i in range(5):
	results = algo.train()
	print(Fore.BLACK + Back.BLUE + f"Iter: {i}, avg_reward = {results['episode_reward_mean']}" + Style.RESET_ALL)



# creating a new environment and state space
# and using the above trained algorithm to solve the corridor traversal problem in it
env = EnergyCorridor({"df": df})
obs = env.reset()
done = False
total_reward = 0.0
while not done:
	# given state = obs, compute action
	action = algo.compute_single_action(obs)
	# take a step given action
	obs, reward, done, info = env.step(action)
	# compute reward
	total_reward += reward

print(Fore.BLACK + Back.GREEN + f"Played 1 episode until done = True, total reward = {total_reward}")






