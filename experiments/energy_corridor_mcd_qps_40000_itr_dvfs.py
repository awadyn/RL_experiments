# EXPERIMENT 0
# dataset: linux mcd logs + 1 qps value = 40000, varying itr-delay and dvfs values
import gym
from ray.rllib.algorithms.ppo import PPO

import pandas as pd
import numpy as np
import utils

# color print
from colorama import Fore, Back, Style
import sys

debug = True
step_count = 0
reset_count = 0



# problem definition: 
#	a corridor in which an agent must move to find minimum energy
#	moves imply changing both ITR-delay and DVFS
class EnergyCorridor(gym.Env):
	def __init__(self, config):
		df = config["df"]
		self.reset_count = 0
		# initializing dictionaries
		state_dict, reward_dict, action_dict, knob_list, key_list = utils.init_dataset(df)	

		# populating dictionaries...
		self.state_space = {}
		self.reward_space = {}
		self.key_space = []
		for key in list(state_dict.keys()):
			self.state_space[key] = state_dict[key]
			self.reward_space[key] = reward_dict[key]
			self.key_space.append(key)
		# N = |observation_space| = |feature_space| = |single_state_vector|
		N = len(state_dict[self.key_space[0]])
		self.observation_space = gym.spaces.Box(low = np.zeros((N)), high = np.inf*np.ones(N))
		# splitting action_dict dictionary into separate itr-delay and dvfs actions 
		self.itr_actions = action_dict['itr']
		self.dvfs_actions = action_dict['dvfs']
		# |action_space| = [3 (increase DVFS, decrease DVFS, or keep unchanged),
		#		    3 (increase ITR-delay, decrease ITR-delay, or keep unchanged)]
		num_actions = [3, 3]
		self.action_space = gym.spaces.MultiDiscrete(num_actions)

		# a simple default reset 
		idx = np.random.randint(len(self.key_space))
		self.cur_key = self.key_space[idx]
		# initializing goal_key to be the key that yields minimum energy for this target_qps
		#self.cur_qps = self.cur_key[2]
		joules = list({k:v['joules_99'] for k,v in self.reward_space.items()}.values())
		# goal key = key that yields min energy
		min_joules = min(joules)
		min_index = joules.index(min_joules)
		self.goal_key = self.key_space[min_index]
		if debug:
			print(Fore.BLACK + Back.RED + "|state_space| =  " + str(len(self.state_space)) + Style.RESET_ALL)
			#print(self.state_space)
			print(Fore.BLACK + Back.RED + "|reward_space| = " + str(len(self.reward_space)) + Style.RESET_ALL)
			#print(self.reward_space)
			print(Fore.BLACK + Back.RED + "|key_space| = " + str(len(self.key_space)) + Style.RESET_ALL)
			#print(self.key_space)
			print(Fore.BLACK + Back.RED + "|action_space| = " + str(len(self.action_space)) + Style.RESET_ALL)
			#print(self.action_space)
		if debug:
			print(Fore.BLACK + Back.RED + "goal_key, min_joules" + Style.RESET_ALL)
			print(self.goal_key, min_joules)
		return



	# runs everytime 1 episode/sequence of many chosen actions is completed
	def reset(self):
		global step_count
		self.reset_count += 1
		if not debug:
			print(Fore.BLACK + Back.RED + "resetting env... step_count = " + str(step_count) + ", reset_count = " + str(self.reset_count) + Style.RESET_ALL)
		step_count = 0

		# a simple randomized reset 
		idx = np.random.randint(len(self.key_space))
		self.cur_key = self.key_space[idx]
		# initializing goal_key to be the key that yields minimum energy for this target_qps
		#self.cur_qps = self.cur_key[2]
		#joules = list({k:v['joules'] for k,v in self.reward_space.items() if k[2] == self.cur_qps}.values())
		#keys = list({k for k in self.key_space if k[2] == self.cur_qps})
		# goal key = key that yields min energy
		#min_joules = min(joules)
		#min_index = joules.index(min_joules)
		#self.goal_key = keys[min_index]
		# returning new state vector
		return self.state_space[self.cur_key]


	def step(self, action):
		global step_count
		#found = 0
		new_key = list(self.cur_key)
		new_itr = self.itr_actions[self.cur_key[0]][action[0] - 1]
		# revert to current itr-delay value if invalid choice
		if new_itr == -1:
			new_itr = self.cur_key[0]
		new_dvfs = self.dvfs_actions[self.cur_key[1]][action[1] - 1] 
		# revert to current dvfs value if invalid choice
		if new_dvfs == -1:
			new_dvfs = self.cur_key[1]
		new_key[0] = new_itr
		new_key[1] = new_dvfs
		new_key = tuple(new_key)
		# check if new key exists
		#if (new_key not in self.key_space):
		#	print("KEY NOT FOUND: ", new_key)
		#	return self.state_space[self.cur_key], 0, False, {'error': 1} 
		new_energy = self.reward_space[new_key]['joules_99']
		done = (new_key == self.goal_key)
		#reward = 1.0 if done else -0.5 if bad move, +0.1 if good move
		reward = 0
		if (self.reward_space[new_key]['joules_99'] < self.reward_space[self.cur_key]['joules_99']):
			reward += 0.1
		else:
			reward -= 0.5
		if done:
			print(Fore.BLACK + Back.RED + "FOUND MIN ENERGY" + Style.RESET_ALL)
			reward += 1

		if not debug:
			print(Fore.BLACK + Back.GREEN + "STEP: action =  " + str(action - 1) + ", reward = " + str(reward) + ", done = " + str(done) + Style.RESET_ALL)
			print(Fore.BLACK + Back.GREEN + "new key: " + Style.RESET_ALL)
			print(new_key)
			print(Fore.BLACK + Back.GREEN + "new energy: " + Style.RESET_ALL)
			print(new_energy)

		step_count += 1
		new_state = self.state_space[new_key]
		return new_state, reward, done, {'error': 0}


featurized_logs_file = sys.argv[1]
df = pd.read_csv(featurized_logs_file, sep = ' ')
# choosing only logs with rapl value 135
#df = df[df['rapl'] == 135]
# choosing only logs with target_qps value 40000
#df = df[df['target_qps'] == 40000]
## normalizing all feature vectors
df = utils.normalize(df)


if debug:
	print()
	print()
	print('----------------------------------------------------------')
	print('FEATURES: ------------------------------------------------')
	print('----------------------------------------------------------')
	print(Fore.BLACK + Back.GREEN + "df: " + Style.RESET_ALL)
	print(df)
	print()
	print()


# defining the training algorithm
# PPO is the mathematical model
algo = PPO(
	config = {
		"env": EnergyCorridor,
		"env_config": {
			"df": df,
		},
		"num_workers": 1,
		"horizon": 15,
	}
)

for i in range(50):
	results = algo.train()
	print(Fore.BLACK + Back.BLUE + f"Iter: {i}, avg_reward = {results['episode_reward_mean']}" + Style.RESET_ALL)


## creating a new environment and state space
## and using the above trained algorithm to solve the corridor traversal problem in it
#env = EnergyCorridor({"df": df})
#obs = env.reset()
#done = False
#finish_count = 0
#total_reward = 0.0
#while not done:
#	# given state = obs, compute action
#	action = algo.compute_single_action(obs)
#	# take a step given action
#	obs, reward, done, info = env.step(action)
#	# NOTE: in case a bad action is chosen, just repeat and choose another
#	if (info['error'] == 1):
#		continue
#	# compute reward
#	total_reward += reward
#	if (finish_count == 1000):
#		print(Fore.BLACK + Back.RED + "Terminated attempt.. done = False")
#		done = True
#	finish_count += 1
#
#print(Fore.BLACK + Back.GREEN + f"Played 1 episode until done = True, total reward = {total_reward}")






