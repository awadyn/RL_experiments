# EXPERIMENT 3
# same as exp 2, but variable itr added

import gym
from ray.rllib.algorithms.ppo import PPO

import pandas as pd
import numpy as np
import utils

# color print
from colorama import Fore, Back, Style

debug = True
step_count = 0
reset_count = 0


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





# problem definition: 
#	a corridor in which an agent must move right or left to find minimum energy
#	moves imply changing both ITR-delay and DVFS
class EnergyCorridor(gym.Env):
	def __init__(self, config):
		df = config["df"]

		self.reset_count = 0

		state_dict, reward_dict, action_dict, knob_list, key_list = utils.init_linux_mcd_dataset(df)
		# for the purpose of this experiment - changing only dvfs
		self.itr_actions = action_dict['itr']
		self.dvfs_actions = action_dict['dvfs']

		# dictionaries of X log encodings with ITR-delay = 100 and QPS = 40000
		self.state_space = {}
		self.reward_space = {}
		self.key_space = []
		for key in list(state_dict.keys()):
			#if key[2] == 40000:
			self.state_space[key] = state_dict[key]
			self.reward_space[key] = reward_dict[key]
			self.key_space.append(key)

		# |observation_space| = as many normalized feature vectors as is available
		#feature_list = list(state_dict[self.key_space[0]].keys())
		N = len(state_dict[self.key_space[0]])
		self.observation_space = gym.spaces.Box(low = np.zeros((N)), high = np.inf*np.ones(N))

		# goal key = key that yields min energy
		joules_99 = list({k:v['joules_99'] for k,v in self.reward_space.items()}.values())
		min_joules_99 = min(joules_99)
		self.goal_key = self.key_space[joules_99.index(min_joules_99)]
		# default start key
		self.cur_key = self.key_space[0]

		# |action_space| = [3 (increase DVFS, decrease DVFS, or keep unchanged),
		#		    3 (increase ITR-delay, decrease ITR-delay, or keep unchanged)]
		num_actions = [3, 3]
		self.action_space = gym.spaces.MultiDiscrete(num_actions)

		if debug:
			print(Fore.BLACK + Back.RED + "|state_space| =  " + str(len(self.state_space)) + Style.RESET_ALL)
			#print(self.state_space)
			print(Fore.BLACK + Back.RED + "|reward_space| = " + str(len(self.reward_space)) + Style.RESET_ALL)
			#print(self.reward_space)
			print(Fore.BLACK + Back.RED + "|key_space| = " + str(len(self.key_space)) + Style.RESET_ALL)
			print(self.key_space)
			print(Fore.BLACK + Back.RED + "|action_space| = " + str(len(self.action_space)) + Style.RESET_ALL)
			print(self.action_space)
			#print(Fore.BLACK + Back.RED + "|observation_space| = " + str(len(self.observation_space)) + Style.RESET_ALL)
			#print(self.observation_space)
		if debug:
			print(Fore.BLACK + Back.RED + "goal_key, min_joules_99" + Style.RESET_ALL)
			print(self.goal_key, min_joules_99)



	# runs everytime 1 episode of many attempted moves is completed
	def reset(self):
		global step_count

		self.reset_count += 1
		if not debug:
			print(Fore.BLACK + Back.RED + "resetting env... step_count = " + str(step_count) + ", reset_count = " + str(self.reset_count) + Style.RESET_ALL)
		step_count = 0

		# a simple default reset 
		idx = np.random.randint(len(self.key_space))
		self.cur_key = self.key_space[idx]

		return self.state_space[self.cur_key]


	def step(self, action):
		global step_count

		new_key = list(self.cur_key)
		new_itr = self.itr_actions[self.cur_key[0]][action[0] - 1]
		if new_itr == -1:
			new_itr = self.cur_key[0]
		new_dvfs = self.dvfs_actions[self.cur_key[1]][action[1] - 1] 
		if new_dvfs == -1:
			new_dvfs = self.cur_key[1]
		new_key[0] = new_itr
		new_key[1] = new_dvfs
		new_key = tuple(new_key)
		new_energy = self.reward_space[new_key]['joules_99']

		done = (new_key == self.goal_key)

		#reward = 1.0 if done else -0.1
		reward = 0
		#diff = self.reward_space[new_key]['joules_99'] < self.reward_space[self.cur_key]['joules_99'] 
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

		return new_state, reward, done, {}



df = pd.read_csv('./features/linux_mcd_features.csv')
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
		"horizon": 10,
	}
)

for i in range(50):
	results = algo.train()
	print(Fore.BLACK + Back.BLUE + f"Iter: {i}, avg_reward = {results['episode_reward_mean']}" + Style.RESET_ALL)


# creating a new environment and state space
# and using the above trained algorithm to solve the corridor traversal problem in it
env = EnergyCorridor({"df": df})
obs = env.reset()
done = False
finish_count = 0
total_reward = 0.0
while not done:
	# given state = obs, compute action
	action = algo.compute_single_action(obs)
	# take a step given action
	obs, reward, done, info = env.step(action)
	# compute reward
	total_reward += reward
	if (finish_count == 1000):
		print(Fore.BLACK + Back.RED + "Terminated attempt.. done = False")
		done = True
	finish_count += 1

print(Fore.BLACK + Back.GREEN + f"Played 1 episode until done = True, total reward = {total_reward}")






