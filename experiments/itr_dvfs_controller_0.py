# CONTROLLER V0
# dataset: linux mcd logs, 1 qps, varying itr-delay and dvfs
import gym
#from ray.rllib.algorithms.ppo import PPO

import plotly.graph_objects as go

#import pandas as pd
import numpy as np
import itertools

import utils

# color print
from colorama import Fore, Back, Style
#import sys

# visualization plots
import plotly.graph_objects as go


debug = True



class EnergyCorridor(gym.Env):

	def __init__(self, config):

		self.training = True
		self.testing = False

		df = config["df"]
		self.step_count = config["step_count"]
		self.reset_count = config["reset_count"]
		self.success_count = config["success_count"]

		

		self.state_space, self.reward_space, action_dict, knob_list, key_set, self.qps = utils.init_dataset(df)	
		key_set, self.state_space, self.reward_space = utils.assert_keys(key_set, self.state_space, self.reward_space, action_dict)

		self.key_space = list(key_set)
		self.itr_actions = action_dict['itr']
		self.dvfs_actions = action_dict['dvfs']

		self.itrs = []
		self.dvfss = []
		self.itrs_visited = []
		self.dvfss_visited = []
		for key in self.key_space:
			key = list(key)
			self.itrs.append(key[0])
			self.dvfss.append(key[1])

		N = len(self.state_space[self.key_space[0]])
		self.observation_space = gym.spaces.Box(low = np.zeros((N)), high = np.inf*np.ones(N))

		num_actions = [3, 3]
		self.action_space = gym.spaces.MultiDiscrete(num_actions)

		joules = list(self.reward_space.values())
		self.goal_energy = min(joules)[0]
		min_index = joules.index(self.goal_energy)
		self.goal_key = list(self.reward_space.keys())[min_index]

		if self.training:
			print(Fore.BLACK + Back.RED + "|state_space| =  " + str(len(self.state_space)) + "    state_space[key_0]: " + Style.RESET_ALL)
			print(self.state_space[self.key_space[0]])
			print(Fore.BLACK + Back.RED + "|reward_space| = " + str(len(self.reward_space)) + "    reward_space[key_0]: " + Style.RESET_ALL)
			print(self.reward_space[self.key_space[0]])
			print(Fore.BLACK + Back.RED + "|key_space| = " + str(len(self.key_space)) + "    key_space: "+ Style.RESET_ALL)
			print(self.key_space)
			print(Fore.BLACK + Back.RED + "|action_space| = " + str(len(self.action_space)) + Style.RESET_ALL)
			print(self.action_space)
			print(Fore.BLACK + Back.RED + "goal_energy = " + str(self.goal_energy) + Style.RESET_ALL)
			print(Fore.BLACK + Back.RED + "goal_key = " + Style.RESET_ALL)
			print(self.goal_key)


		# a simple default reset 
		idx = np.random.randint(len(self.key_space))
		self.cur_key = self.key_space[idx]
		self.episode_reward = 0
		return


	def reset(self, freq = 20, start_key=None):

		if (self.training and (self.reset_count % freq == 0)):
			print(Fore.BLACK + Back.CYAN + "reset_count =  " + str(self.reset_count) + "    success_count = " + str(self.success_count) + Style.RESET_ALL)

		self.episode_reward = 0
		self.step_count = 0
		self.reset_count += 1

		if self.testing:
			self.cur_key = start_key
			self.itrs_visited = []
			self.dvfss_visited = []
			self.itrs_visited.append(list(self.cur_key)[0])
			self.dvfss_visited.append(list(self.cur_key)[1])

		else:
			idx = np.random.randint(len(self.key_space))
			self.cur_key = self.key_space[idx]

		return self.state_space[self.cur_key]


	def step(self, action):
		
		self.step_count += 1

		new_key = list(self.cur_key)
		new_itr = self.itr_actions[self.cur_key[0]][action[0] - 1]
		if new_itr == -1:
			new_itr = self.cur_key[0]
		new_dvfs = self.dvfs_actions[self.cur_key[1]][action[1] - 1] 
		if new_dvfs == -1:
			new_dvfs = self.cur_key[1]

		if self.testing:
			self.itrs_visited.append(new_itr)
			self.dvfss_visited.append(new_dvfs)

		new_key[0] = new_itr
		new_key[1] = new_dvfs
		new_key = tuple(new_key)

		new_energy = self.reward_space[new_key][0]
		diff_energy = self.reward_space[self.cur_key][0] - self.reward_space[new_key][0]
		done = (new_energy == self.goal_energy)
		reward = 0
		if (diff_energy > 0):
			reward += diff_energy	
		else:
			reward += 5 * diff_energy
		if done:
			self.success_count += 1

		#if self.testing:
			#if (done):
			#	print(Fore.CYAN + Back.BLACK + "STEP:    action =  " + str(action - 1) + "    done = " + str(done) + "    new_energy: " + str(new_energy) + Style.RESET_ALL)
			#k0_0 = str(self.cur_key[0])
			#k0_1 = str(self.cur_key[1])
			#k1_0 = str(new_key[0])
			#k1_1 = str(new_key[1])
			#print(Fore.CYAN + Back.BLACK + "STEP:    action =  " + str(action - 1) + "   ( " + k0_0 + ", " + k0_1 + " ) -->    ( " + k1_0 + ", " + k1_1 + " )" + "    reward = " + str(reward) + "    done = " + str(done) + "    new_energy: " + str(new_energy) + Style.RESET_ALL)
			#print(Fore.CYAN + Back.BLACK + "STEP:    action =  " + str(action - 1) +  "    reward = " + str(reward) + "    done = " + str(done) + "    new_energy: " + str(new_energy) + Style.RESET_ALL)
		
		self.cur_key = new_key
		new_state = self.state_space[new_key]
		self.episode_reward += reward
		return new_state, reward, done, {'error': 0, 'key': new_key}




#featurized_logs_file = sys.argv[1]
#featurized_logs_test_file = sys.argv[2]
#df = pd.read_csv(featurized_logs_file, sep = ',')
#df_test = pd.read_csv(featurized_logs_test_file, sep = ',')
#
#if debug:
#	print()
#	print()
#	print(Fore.BLACK + Back.GREEN + "df: " + Style.RESET_ALL)
#	print(df)
#	print()
#	print()
#	print(Fore.BLACK + Back.GREEN + "df_test: " + Style.RESET_ALL)
#	print(df_test)
#	print()
#	print()
#
#
#algo = PPO(
#	config = {
#		"env": EnergyCorridor,
#		"env_config": {
#			"df": df,
#			"reset_count": 0,
#			"success_count": 0,
#			"step_count": 0,
#		},
#		"framework": 'torch',
#		"num_workers": 1,
#		"horizon": 20,
#		"seed": 4,
#		"gamma": 0.5,
##		"lr": 1e-6,
#	}
#)
#
## test seed
## [np.random.random() for i in range(10)]
#def fix_seeds(seed=0):
#	pass
##	np.random.seed(seed)
##	torch.manual_seed(seed)
##	torch.use_deterministic_algorithms(True)
#
#for i in range(25):
#	results = algo.train()
#	print(Fore.BLACK + Back.BLUE + f"Iter: {i}, episode_reward_mean = {results['episode_reward_mean']}" + Style.RESET_ALL)
#
#checkpoint_file = algo.save("trained_models/RL_model_0_gamma_9")




########################################################################################

#env = EnergyCorridor({"df": df_test, "step_count": 0, "reset_count": 0, "success_count": 0})
#
#env.training = False
#env.testing = True
#
#fig = go.Figure()
#plot_dvfss = []
#plot_itrs = []
#plot_rewards = []
#for (i, d) in zip(env.itrs, env.dvfss):
#	if d < 10000:
#		plot_dvfss.append(d)
#		plot_itrs.append(i)
#		plot_rewards.append(env.reward_space[(i,d)])
#fig.add_trace(go.Scatter(x=plot_itrs, y=plot_dvfss, text=plot_rewards, mode='markers'))
#fig.add_trace(go.Scatter(x=[list(env.goal_key)[0]], y=[list(env.goal_key)[1]], marker_size=20, marker_color = "yellow"))	
#fig.update_layout(title="QPS: " + str(env.qps) + " - Gamma: " + str(algo.config['gamma']) + " - lr: " + str(algo.config['lr']))
#
#for key in env.key_space:
#	obs = env.reset(start_key = key)
#	done = False
#	for i in range(20):
#		action = algo.compute_single_action(obs)
#		obs, reward, done, info = env.step(action)
#
#		if done:
#			break
#
#	trace_name = str(key[0]) + " , " + str(key[1])
#	if key[1] < 10000:
#		if not done:
#			fig.add_trace(go.Scatter(x=env.itrs_visited, y=env.dvfss_visited, name=trace_name, marker= dict(size=10,symbol= "arrow-bar-up", angleref="previous")))
#			print()
#			print("NOT DONE")
#			print()
#		else:
#			fig.add_trace(go.Scatter(x=env.itrs_visited, y=env.dvfss_visited, name=trace_name, marker= dict(size=10,symbol= "arrow-bar-up", angleref="previous"), marker_color="black"))
#	print("itrs_visited: ", env.itrs_visited)
#	print("dvfss_visited: ", env.dvfss_visited)
#
#
#
#fig.show()


