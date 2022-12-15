# EXPERIMENT 0
# dataset: linux mcd logs + 1 qps value, varying itr-delay and dvfs values
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
		state_dict, reward_dict, action_dict, knob_list, key_set = utils.init_dataset(df)	

		# populating dictionaries...
		self.state_space = {}
		self.reward_space = {}
		self.key_space = []
		self.sorted_keys = []
		for key in key_set:
			#numeric_key = tuple([int(list(key)[0]), int(list(key)[1], base=16)])
			self.state_space[key] = state_dict[key]
			self.reward_space[key] = reward_dict[key]
			self.key_space.append(key)
		self.sorted_keys = sorted(self.key_space)

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

		if debug:
			print(Fore.BLACK + Back.RED + "|state_space| =  " + str(len(self.state_space)) + Style.RESET_ALL)
			print(self.state_space[self.key_space[0]], '...')
			#print(self.state_space.items())
			print(Fore.BLACK + Back.RED + "|reward_space| = " + str(len(self.reward_space)) + Style.RESET_ALL)
			print(self.reward_space[self.key_space[0]], '...')
			#print(self.reward_space.items())
			print(Fore.BLACK + Back.RED + "|key_space| = " + str(len(self.key_space)) + Style.RESET_ALL)
			print(self.key_space)
			print(Fore.BLACK + Back.RED + "|action_space| = " + str(len(self.action_space)) + Style.RESET_ALL)
			print(self.action_space)

		# a simple default reset 
		idx = np.random.randint(len(self.key_space))
		self.cur_key = self.key_space[idx]

		# initializing goal_key to be the key that yields minimum energy for this target_qps
		joules = list({k:v[0] for k,v in self.reward_space.items()}.values())
		# goal key = key that yields min energy
		min_joules = min(joules)
		self.goal_energy = min_joules
		#min_index = joules.index(min_joules)
		#self.goal_key = self.key_space[min_index]
		if debug:
			print(Fore.BLACK + Back.RED + "goal_energy = " + Style.RESET_ALL)
			print(self.goal_energy)
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

		reward = 0
		while True:
			try:
				new_energy = self.reward_space[new_key][0]
				diff_energy = np.abs(self.reward_space[self.cur_key][0] - self.reward_space[new_key][0])
				done = (new_energy == self.goal_energy)
				#reward = 1.0 if done else -0.5 if bad move, +0.1 if good move
				if (new_energy < self.reward_space[self.cur_key][0]):
					reward += 0.5
					#reward += 5 * diff_energy
				else:
					reward -= 0.1
					#reward -= diff_energy
				if done:
					print(Fore.BLACK + Back.RED + "FOUND MIN ENERGY" + Style.RESET_ALL)
					reward += 1
				if debug:
					print(Fore.BLACK + Back.GREEN + "STEP: action =  " + str(action - 1) + ", reward = " + str(reward) + ", done = " + str(done) + Style.RESET_ALL)
					print(Fore.BLACK + Back.GREEN + "new key: " + Style.RESET_ALL)
					print(new_key)
					print(Fore.BLACK + Back.GREEN + "new energy: " + Style.RESET_ALL)
					print(new_energy)
				break
			except KeyError:
				print(Fore.BLACK + Back.YELLOW + "Stepping toward " + str(new_key) + "... This key is not found... try interpolation to fillin missing log data." + Style.RESET_ALL)
				target_states = []
				target_rewards = []
				if (action[0] - 1 == 0):
					print("only dvfs changed...")
					# find keys with target dvfs value
					target_keys = [key for key in self.sorted_keys if key[1] == new_key[1]]
					print("target_keys: ", target_keys)
					# new state will be the mean of target key states
					for key in target_keys:
						target_states.append(self.state_space[key])
						target_rewards.append(self.reward_space[key])
					target_states.append(self.state_space[self.cur_key])
					target_rewards.append(self.reward_space[self.cur_key])
					print(target_states)
					print(target_rewards)
					new_state = np.mean(target_states, axis=0)
					new_reward = np.mean(target_rewards, axis=0)
					print("new_state = ", new_state)
					print("new_reward = ", new_reward)
					# add interpolated state and reward to environment
					self.state_space[new_key] = new_state
					self.reward_space[new_key] = new_reward
					print()
					print(self.state_space[new_key], self.reward_space[new_key])
					print()
				elif (action[1] - 1 == 0):
					print("only itr-delay changed...")
					# find keys with target itr-delay value
					target_keys = [key for key in self.sorted_keys if key[0] == new_key[0]]
					# interpolate by averaging current key state and target key state
					print("target_keys: ", target_keys)
					# new state will be the mean of target key states
					for key in target_keys:
						target_states.append(self.state_space[key])
						target_rewards.append(self.reward_space[key])
					target_states.append(self.state_space[self.cur_key])
					target_rewards.append(self.reward_space[self.cur_key])
					print(target_states)
					print(target_rewards)
					new_state = np.mean(target_states, axis=0)
					new_reward = np.mean(target_rewards, axis=0)
					print("new_state = ", new_state)
					print("new_reward = ", new_reward)
					# add interpolated state to state_space
					self.state_space[new_key] = new_state
					self.reward_space[new_key] = new_reward
					print()
					print(self.state_space[new_key], self.reward_space[new_key])
					print()
				else:
					print("both itr-delay and dvfs changed...")
					# find keys with target itr-delay value
					target_keys = [key for key in self.sorted_keys if key[0] == new_key[0]]
					# find keys with target dvfs value and current itr-delay value
					for key in self.sorted_keys:
						if (key[1] == new_key[1] and key[0] == self.cur_key[0]):
							target_keys.append(key)
					print("target_keys: ", target_keys)
					# new state will be the mean of target key states
					for key in target_keys:
						target_states.append(self.state_space[key])
						target_rewards.append(self.reward_space[key])
					target_states.append(self.state_space[self.cur_key])
					target_rewards.append(self.reward_space[self.cur_key])
					print(target_states)
					print(target_rewards)
					new_state = np.mean(target_states, axis=0)
					new_reward = np.mean(target_rewards, axis=0)
					print("new_state = ", new_state)
					print("new_reward = ", new_reward)
					# add interpolated state to state_space
					self.state_space[new_key] = new_state
					self.reward_space[new_key] = new_reward
					print()
					print(self.state_space[new_key], self.reward_space[new_key])
					print()
					
				continue		
	
		step_count += 1
		new_state = self.state_space[new_key]
		return new_state, reward, done, {'error': 0}


featurized_logs_file = sys.argv[1]
df = pd.read_csv(featurized_logs_file, index_col=0, sep = ' ')
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
		"framework": 'torch',
		"num_workers": 1,
		"horizon": 20,
	}
)

for i in range(200):
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






