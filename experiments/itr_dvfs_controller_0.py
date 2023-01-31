# CONTROLLER V0
# dataset: linux mcd logs, 1 qps, varying itr-delay and dvfs
import gym
from ray.rllib.algorithms.ppo import PPO

import plotly.graph_objects as go

import pandas as pd
import numpy as np
import utils

# color print
from colorama import Fore, Back, Style
import sys

# visualization plots
import plotly.graph_objects as go


debug = True



def init_dataset(df):

	energy_cols = ['joules_99']
	id_cols = ['core', 'sys', 'exp']
	knob_cols = ['itr', 'dvfs']
	skip_cols = ['fname']
	reward_cols = ['core', 'sys', 'exp', 'joules_99']
	
	for col in df.drop(['fname', 'sys' , 'core', 'exp', 'itr', 'dvfs'], axis = 1).columns:
		# sanity check
		if (df[col].max() - df[col].min() == 0):
			continue
		df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

	df_state = df.set_index(knob_cols).drop(energy_cols, axis=1).drop(skip_cols, axis=1)
	df_reward = df.set_index(knob_cols)[reward_cols]

	df_state = df_state.sort_index()
	df_reward = df_reward.sort_index()

	key_list = list(df_state.index)
	key_set = set(key_list)
	if debug:
		print(Fore.BLACK + Back.GREEN + "|key_list| = " + str(len(key_list)) + "    key_list = ..." + Style.RESET_ALL)
		print(Fore.BLACK + Back.BLUE + "|key_set| = " + str(len(key_set)) + "    key_set = " + Style.RESET_ALL)
		print(key_set)

	state_dict = {}
	reward_dict = {}
	for key in key_set:
		states_per_key = df_state.loc[key].drop(id_cols, axis=1)
		rewards_per_key = df_reward.loc[key].drop(id_cols, axis=1)
		# each key should be paired with 16 execution states, one from each core
		num_reps = len(states_per_key)
		avg_state_per_key = np.add.reduce(states_per_key.values)/num_reps
		avg_reward_per_key = np.add.reduce(rewards_per_key.values)/num_reps
		state_dict[key] = np.array(list(avg_state_per_key))
		reward_dict[key] = np.array(list(avg_reward_per_key))

	action_dict, knob_list = prepare_action_dicts(df)

	if debug:
		print(Fore.BLACK + Back.GREEN + "|state_dict| = " + str(len(state_dict.items())) + "    state_dict[key_0]: " + Style.RESET_ALL)
		print(list(state_dict.items())[0:1])
		print(Fore.BLACK + Back.GREEN + "|reward_dict| = " + str(len(reward_dict.items())) + "    reward_dict[key_0]: " + Style.RESET_ALL)
		print(list(reward_dict.items())[0:1])
		print(Fore.BLACK + Back.GREEN + "knob_list: " + Style.RESET_ALL)
		print(knob_list)
		print(Fore.BLACK + Back.GREEN + "action_dict: " + Style.RESET_ALL)
		print(action_dict)

	return state_dict, reward_dict, action_dict, knob_list, key_set



def prepare_action_dicts(df):
	def get_knob_dict(knob):
		l = np.sort(df[knob].unique())
		l_p1 = np.roll(l, shift=-1)
		l_p1[-1] = -1 #invalid choice
		l_m1 = np.roll(l, shift=1)
		l_m1[0] = -1 #invalid choice
		d = {}
		for idx, elem in enumerate(l):

			pre = l_m1[idx]
			suc = l_p1[idx]
			d[elem] = {-1: pre, 0: elem, 1: suc}
		return d

	d = {}
	knob_list = []
	for knob in ['itr', 'dvfs']:
		knob_list.append(knob)
		d[knob] = get_knob_dict(knob)	

	return d, knob_list





class EnergyCorridor(gym.Env):

	def __init__(self, config):

		self.training = True
		self.testing = False

		df = config["df"]
		self.step_count = config["step_count"]
		self.reset_count = config["reset_count"]
		self.success_count = config["success_count"]

		self.state_space, self.reward_space, action_dict, knob_list, key_set = init_dataset(df)	
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

		joules = list({k:v[0] for k,v in self.reward_space.items()}.values())
		min_joules = min(joules)
		self.goal_energy = min_joules
		min_index = joules.index(min_joules)
		self.goal_key = self.key_space[min_index]

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


	def reset(self, freq = 10):

		if (self.training and (self.reset_count % 10 == 0)):
			print(Fore.BLACK + Back.CYAN + "reset_count =  " + str(self.reset_count) + "    success_count = " + str(self.success_count) + Style.RESET_ALL)

		self.episode_reward = 0
		self.step_count = 0
		self.reset_count += 1

		idx = np.random.randint(len(self.key_space))
		self.cur_key = self.key_space[idx]

		if self.testing:
			self.itrs_visited = []
			self.dvfss_visited = []
			self.itrs_visited.append(list(self.cur_key)[0])
			self.dvfss_visited.append(list(self.cur_key)[1])

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
			reward += 10
		else:
			reward -= 50
		if done:
			self.success_count += 1

		if self.testing:
			print(Fore.CYAN + Back.BLACK + "STEP:    action =  " + str(action - 1) + "    reward = " + str(reward) + "    done = " + str(done) + "    cur_key, new key, new_energy: " + Style.RESET_ALL)
			print(self.cur_key, new_key, new_energy)
		
		self.cur_key = new_key
		new_state = self.state_space[new_key]
		self.episode_reward += reward
		return new_state, reward, done, {'error': 0, 'key': new_key}




featurized_logs_file = sys.argv[1]
df = pd.read_csv(featurized_logs_file, sep = ',')
D = int(len(df)/3)
df_train = df[0:2*D]
df_test = df[2*D+1:]

if debug:
	print()
	print()
	print(Fore.BLACK + Back.GREEN + "df: " + Style.RESET_ALL)
	print(df)
	print()
	print()


algo = PPO(
	config = {
		"env": EnergyCorridor,
		"env_config": {
			"df": df_train,
			"reset_count": 0,
			"success_count": 0,
			"step_count": 0,
		},
		"framework": 'torch',
		"num_workers": 1,
		"horizon": 12,
		#"gamma": 0.9,
		#"lr": 1e-4,
	}
)



for i in range(20):
	results = algo.train()
	print(Fore.BLACK + Back.BLUE + f"Iter: {i}, episode_reward_mean = {results['episode_reward_mean']}" + Style.RESET_ALL)



env = EnergyCorridor({"df": df_test, "step_count": 0, "reset_count": 0, "success_count": 0})
env.training = False
env.testing = True
fig = go.Figure()
fig.add_trace(go.Scatter(x=env.itrs, y=env.dvfss, mode='markers'))
all_itrs_visited = []
all_dvfss_visited = []
for i in range(50):
	print("resetting env...")
	print()
	obs = env.reset()
	done = False
	for i in range(15):
		action = algo.compute_single_action(obs)
		obs, reward, done, info = env.step(action)

		if done:
			break
	print("itrs_visited: ", env.itrs_visited, "   dvfss_visited: ", env.dvfss_visited)
	all_itrs_visited.append(env.itrs_visited)
	all_dvfss_visited.append(env.dvfss_visited)



fig.add_trace(go.Scatter(x=[list(env.goal_key)[0]], y=[list(env.goal_key)[1]], marker_size=20, marker_color = "yellow"))	
for i in range(50):
	fig.add_trace(go.Scatter(x=all_itrs_visited[i], y=all_dvfss_visited[i], marker= dict(size=10,symbol= "arrow-bar-up", angleref="previous")))

fig.show()


