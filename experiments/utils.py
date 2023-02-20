import numpy as np
import re
# color print
from colorama import Fore, Back, Style

debug = True

def init_dataset(df):

	itrs = []
	dvfss = []
	qpss = []
	runs = []
	cores = []	
	# adding itr, dvfs, and qps columns to df
	for k, v in df.iterrows():
		f = v['fname']
		run = int(re.search(r'linux\.mcd\.dmesg\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(1))
		core = int(re.search(r'linux\.mcd\.dmesg\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(2))
		itr = int(re.search(r'linux\.mcd\.dmesg\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(3))
		dvfs = int(re.search(r'linux\.mcd\.dmesg\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(4), base=16)
		qps = int(re.search(r'linux\.mcd\.dmesg\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(6))
		itrs.append(itr)
		dvfss.append(dvfs)
		qpss.append(qps)
		runs.append(run)
		cores.append(core)
	df['itr'] = itrs
	df['dvfs'] = dvfss
	df['qps'] = qpss
	df['run'] = runs
	df['core'] = cores

	energy_cols = ['joules_sum', 'joules_1', 'joules_10', 'joules_25', 'joules_50', 'joules_75', 'joules_90', 'joules_99', 'joules_per_interrupt']
	id_cols = ['core', 'run']
	knob_cols = ['itr', 'dvfs']
	skip_cols = ['fname', 'qps']
	reward_cols = ['core', 'run', 'joules_sum']
	
	non_norm_cols = np.concatenate((energy_cols, id_cols, knob_cols, skip_cols), axis = 0) 
	# normalize columns
	for col in df.drop(non_norm_cols, axis = 1).columns:
		# sanity check
		if (df[col].max() - df[col].min() == 0):
			continue
		df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
		break

	df_state = df.set_index(knob_cols).drop(energy_cols, axis=1).drop(skip_cols, axis=1)
	df_reward = df.set_index(knob_cols)[reward_cols]

	df_state = df_state.sort_index()
	df_reward = df_reward.sort_index()
	state_dict = {}
	reward_dict = {}

	key_list = list(df_state.index)
	key_set = set(key_list)
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
		print(Fore.BLACK + Back.GREEN + "|key_list| = " + str(len(key_list)) + "    key_list = ..." + Style.RESET_ALL)
		print(Fore.BLACK + Back.BLUE + "|key_set| = " + str(len(key_set)) + "    key_set = " + Style.RESET_ALL)
		print(key_set)
		print(Fore.BLACK + Back.GREEN + "|state_dict| = " + str(len(state_dict.items())) + "    state_dict[key_0]: " + Style.RESET_ALL)
		print(list(state_dict.items())[0:1])
		print(Fore.BLACK + Back.GREEN + "|reward_dict| = " + str(len(reward_dict.items())) + "    reward_dict[key_0]: " + Style.RESET_ALL)
		print(list(reward_dict.items())[0:1])
		print(Fore.BLACK + Back.GREEN + "knob_list: " + Style.RESET_ALL)
		print(knob_list)
		print(Fore.BLACK + Back.GREEN + "action_dict: " + Style.RESET_ALL)
		print(action_dict)

	return state_dict, reward_dict, action_dict, knob_list, key_set, qpss[0]



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


# checks dataset for presence of all keys that the RL agent may visit
# if a key is not found, appends a new entry to state space and reward space
#	with this key, where new state = state avg of nearest neighboring keys
#	and similarly for new reward
def assert_keys(key_set, state_space, reward_space, action_dict):
	itr_actions = action_dict['itr']
	dvfs_actions = action_dict['dvfs']

	key_set_new = list(key_set)
	state_space_new = state_space
	reward_space_new = reward_space

	# for each key, try all possible 9 actions
	for cur_key in key_set_new:
		itr = list(cur_key)[0]
		dvfs = list(cur_key)[1]
		for itr_action in [-1, 0, 1]:
			new_itr = itr_actions[itr][itr_action]
			for dvfs_action in [-1, 0, 1]:
				new_dvfs = dvfs_actions[dvfs][dvfs_action]
				if new_itr == -1:
					new_itr = cur_key[0]
				if new_dvfs == -1:
					new_dvfs = cur_key[1]
				new_key = tuple([new_itr, new_dvfs])

				# handle missing key cases
				if new_key not in key_set_new:
					if debug:
						print(Fore.BLACK + Back.YELLOW + "Missing key: " + Style.RESET_ALL)
						print(new_key)

					# find euclidean distances of all keys from this key
					distances = {}
					nk = np.array(list(new_key))
					for key in key_set_new:
						distances[key] = np.linalg.norm(np.array(list(key)) - nk)
					# sort distances
					sorted_distances = sorted(distances.items(), key=lambda x:x[1])
					# use closest 2 keys as inputs to interpolation
					target_keys = [sorted_distances[0][0], sorted_distances[1][0]]
					if debug:
						print("nearest neighbor keys: ", target_keys)

					target_states = []
					target_rewards = []
					for key in target_keys:
						target_states.append(state_space_new[key])
						target_rewards.append(reward_space_new[key])
					target_states.append(state_space_new[cur_key])
					target_rewards.append(reward_space_new[cur_key])
					# new state and reward will be the mean of target key states
					new_state = np.mean(target_states, axis=0)
					new_reward = np.mean(target_rewards, axis=0)
					if debug:
						print("new_key = ", new_key)
						print("new_state = ", new_state)
						print("new_reward = ", new_reward)

					# add interpolated state and reward to environment
					key_set_new.append(new_key)
					state_space_new[new_key] = new_state
					reward_space_new[new_key] = new_reward

	return set(key_set_new), state_space_new, reward_space_new 

