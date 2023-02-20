import os
import glob
import numpy as np
import pandas as pd
# color print
from colorama import Fore, Back, Style
import warnings


debug = True

def get_params(loc, tag='linux.mcd.dmesg'):
    fnames = glob.glob(f'{loc}/*')

    tag_dict = {'core': [],
                'run': [],
                'itr': [],
                'dvfs': [],
                'rapl': [],
                'qps': []}

    fname_dict = {}

    for f in fnames:
        fields = f.split('/')[-1].split('.')[-1].split('_')
        if len(fields)==6: #excluding default
            run, core, itr, dvfs, rapl, qps = fields

            tag_dict['run'].append(run)
            tag_dict['core'].append(core)
            tag_dict['itr'].append(itr)
            tag_dict['dvfs'].append(dvfs)
            tag_dict['rapl'].append(rapl)
            tag_dict['qps'].append(qps)

            fname_dict[f.split('/')[-1]] = 1

    return tag_dict, fname_dict

def check_grid(tag_dict, fname_dict):
    uniq_itr = np.unique(tag_dict['itr'])
    uniq_dvfs = np.unique(tag_dict['dvfs'])
    uniq_rapl = np.unique(tag_dict['rapl'])
    uniq_qps = np.unique(tag_dict['qps'])
    uniq_cores = np.unique(tag_dict['core'])

    run = 0
    missing_list = []
    for itr in uniq_itr:
        for dvfs in uniq_dvfs:
            for rapl in uniq_rapl:
                for qps in uniq_qps:
                    for core in uniq_cores:
                        fname = f'linux.mcd.dmesg.{run}_{core}_{itr}_{dvfs}_{rapl}_{qps}'
                        if fname not in fname_dict:
                            print(fname)
                            missing_list.append(fname)

    return missing_list

def combine_data(loc):
    #HARDCODED: Fix
    df_list = []

    for f in glob.glob(f'{loc}/linux*.csv'):
        df = pd.read_csv(f)
        df_list.append(df)

    df = pd.concat(df_list, axis=0).reset_index()

    if df.shape[0]==0:
        raise ValueError(f"No data found {loc}")

    #determine workload
    fname = df['fname'][0]
    workload = fname.split('/')[-1].split('.')[1]

    if workload!='mcd':
        raise ValueError(f'Encountered non-mcd workload = {workload}. Ensure logic consistent with new workload.')

    #sys, workload, qps, 
    df['sys'] = df['fname'].apply(lambda x: x.split('/')[-1].split('.')[0])
    df['itr'] = df['fname'].apply(lambda x: int(x.split('/')[-1].split('.')[-1].split('_')[2]))
    df['dvfs'] = df['fname'].apply(lambda x: int(x.split('/')[-1].split('.')[-1].split('_')[3], base=16))
    df['rapl'] = df['fname'].apply(lambda x: int(x.split('/')[-1].split('.')[-1].split('_')[4]))
    df['core'] = df['fname'].apply(lambda x: int(x.split('/')[-1].split('.')[-1].split('_')[1]))
    df['exp'] = df['fname'].apply(lambda x: int(x.split('/')[-1].split('.')[-1].split('_')[0]))

    #remove default policy entries
    df = df[(df['dvfs']!='0xffff') & (df['itr']!=1)].copy()

    if 'index' in df:
        df.drop('index', axis=1, inplace=True)

    return df

def normalize(df):
    col_tags = ['instructions', 
                'cycles', 
                'ref_cycles', 
                'llc_miss', 
                'c1', 
                'c1e', 
                'c3', 
                'c6',
                'c7', 
                'joules',
                'rx_desc',
                'rx_bytes',
                'tx_desc',
                'tx_bytes']

    keep_tags = [c for c in df.columns if '_'.join(c.split('_')[:-1]) not in col_tags]
    process_tags = [c for c in df.columns if '_'.join(c.split('_')[:-1]) in col_tags]

    df_keep = df[keep_tags].copy()
    df_process = df[process_tags].copy()

    df_list = []
    for col in col_tags:
        if col=='c6':
            continue
        cols = [c for c in df.columns if '_'.join(c.split('_')[:-1])==col]
        max_val = df_process[cols].max().max()
        print(cols)
        print(max_val)

        df_list.append(df_process[cols] / max_val)

    df_process = pd.concat(df_list, axis=1)
    df = pd.concat([df_keep, df_process], axis=1)

    return df

def missing_rdtsc_out_files(loc, debug=False):
    
    def check_rdtsc_out_file(fname, debug=False):
        loc = '/'.join(fname.split('/')[:-1])
        tag = fname.split('.')[-1].split('_')
        desc = '_'.join(np.delete(tag, [1]))
        expno = tag[0]

        if debug:
            print(fname) #data/qps_200000/linux.mcd.dmesg.1_10_100_0x1700_135_200000
            print(loc) #data/qps_200000
            print(tag) #['1', '10', '100', '0x1700', '135', '200000']
            print(desc) #1_100_0x1700_135_200000
            print(expno) #1

        rdtsc_fname = f'{loc}/linux.mcd.rdtsc.{desc}' 
        out_fname = f'{loc}/linux.mcd.out.{desc}' 

        rdtsc_found = False
        out_found = False
        
        if os.path.exists(rdtsc_fname):
            rdtsc_found = True
        
        if os.path.exists(out_fname):
            out_found = True

        if not (rdtsc_found and out_found):
            print(f'Missing rdtsc of fname for: {fname}')

    for fname in glob.glob(f'{loc}/linux.mcd.dmesg*'):
        check_rdtsc_out_file(fname, debug=debug)


def init_dataset(df):
	warnings.filterwarnings('ignore')


	#df = df[0:200]

	reward_cols = ['joules_99']		#'joules_per_interrupt', 'time_per_interrupt',
	#reward_cols = ['joules_per_interrupt']
	# a featurized vector is identified by its run id (i.e. run#_core#) and qps id
	#id_cols = ['i']
	id_cols = ['core', 'sys', 'exp']
	# a featurized vector is marked by its itr and dvfs settings
	knob_cols = ['itr', 'dvfs']
	# NOTE: ignoring qps value for now since it is constant
	#skip_cols = ['qps']
	skip_cols = ['fname']

	df_state = df.set_index(knob_cols).drop(reward_cols, axis=1).drop(skip_cols, axis=1)
	df_reward = df.set_index(knob_cols)[reward_cols]

	print("df_state:")
	print(df_state)

	# normalize
	#for col in df_state.drop(id_cols, axis=1).columns:
	#	# sanity check
	#	if (df[col].max() - df[col].min() == 0):
	#		continue
	#	df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
	#for col in df_reward.columns:
	#	# sanity check
	#	if (df[col].max() - df[col].min() == 0):
	#		continue
	#	df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

	key_list = list(df_state.index)
	key_set = set(key_list)

	#bad_keys = []
	#for key in key_set:
	#	if list(key)[0] == 1:
	#		bad_keys.append(key)
	#	if list(key)[1] == '0xffff':
	#		bad_keys.append(key)
	#bad_keys = set(bad_keys)
	#for key in bad_keys:
	#	key_set.remove(key)

	state_dict = {}
	reward_dict = {}
	for key in key_set:
		states_per_key = df_state.loc[key].drop(id_cols, axis=1)
		rewards_per_key = df_reward.loc[key]
		num_reps = len(states_per_key)
		avg_state_per_key = np.add.reduce(states_per_key.values)/num_reps
		avg_reward_per_key = np.add.reduce(rewards_per_key.values)/num_reps
		state_dict[key] = np.array(list(avg_state_per_key))
		reward_dict[key] = np.array(list(avg_reward_per_key))

	#state_dict = df_state.T.to_dict()
	#reward_dict = df_reward.T.to_dict()


	# NOTE: need to do this transform for rllib to not complain
	# NOTE: can probably skip this now
	#for key in state_dict:
	#	state_dict[key] = np.array(list(state_dict[key]))
	#	reward_dict[key] = np.array(list(reward_dict[key]))

	#action_dict, knob_list = prepare_action_dicts(df)
	action_dict, knob_list = prepare_action_dicts(df, key_set)


	if debug:
		print(Fore.BLACK + Back.GREEN + "state_dict: " + Style.RESET_ALL)
		print(len(state_dict.keys()))
		#print(state_dict)
		print(Fore.BLACK + Back.GREEN + "reward_dict: " + Style.RESET_ALL)
		print(len(reward_dict.keys()))
		#print(reward_dict)
		print(Fore.BLACK + Back.GREEN + "action_dict: " + Style.RESET_ALL)
		#print(action_dict)
		print(Fore.BLACK + Back.GREEN + "knob_list: " + Style.RESET_ALL)
		print(knob_list)
		print(Fore.BLACK + Back.GREEN + "key_set: " + Style.RESET_ALL)
		print(key_set)
		#print(len(key_set))		# len = 90
		#print(len(key_list))		# len = 90
		#print(state_dict.keys())

	return state_dict, reward_dict, action_dict, knob_list, key_set


# NOTE: features are average values across all cores of each experiment
# dataset features:
# i sys itr dvfs rapl read_5th read_10th read_50th read_90th read_95th read_99th measure_qps target_qps time joules rx_desc rx_bytes tx_desc tx_bytes instructions cycles ref_cycles llc_miss c1 c1e c3 c6 c7 num_interrupts
def init_compact_dataset(df):
	reward_cols = ['joules', 'time']
	skip_cols = ['i', 'sys', 'rapl', 'measure_qps']
	# state features: read_5th read_10th read_50th read_90th read_95th read_99th rx_desc rx_bytes tx_desc tx_bytes instructions cycles ref_cycles llc_miss c1 c1e c3 c6 c7 num_interrupts  
	df_state = df.set_index(['itr', 'dvfs', 'target_qps']).drop(reward_cols, axis=1).drop(skip_cols, axis = 1)
	df_reward = df.set_index(['itr', 'dvfs', 'target_qps'])[reward_cols]
	#df_state = df.set_index(['itr', 'dvfs']).drop(reward_cols, axis=1).drop(skip_cols, axis = 1)
	#df_reward = df.set_index(['itr', 'dvfs'])[reward_cols]
	state_dict = df_state.T.to_dict()
	# NOTE: need to do this transform for rllib to not complain
	for key in state_dict:
		state_dict[key] = np.array(list(state_dict[key].values()))
	reward_dict = df_reward.T.to_dict()
	key_list = list(state_dict.keys())
	action_dict, knob_list = prepare_action_dicts(df)
	#action_dict, knob_list = prepare_action_dicts(df, key_list)

	if debug:
		print(Fore.BLACK + Back.GREEN + "state_dict: " + Style.RESET_ALL)
		print(df_state)
		print(Fore.BLACK + Back.GREEN + "reward_dict: " + Style.RESET_ALL)
		print(df_reward)
		print(Fore.BLACK + Back.GREEN + "| action_dict | = "+ str(len(list(action_dict['itr'].items())) + len(list(action_dict['dvfs'].items()))) + Style.RESET_ALL)
		print(Fore.BLACK + Back.GREEN + "action_dict: " + Style.RESET_ALL)
		print(action_dict)
		print(Fore.BLACK + Back.GREEN + "knob_list: " + Style.RESET_ALL)
		print(knob_list)
		print(Fore.BLACK + Back.GREEN + "| key_list | = " + str(len(key_list)) + Style.RESET_ALL)
		print(Fore.BLACK + Back.GREEN + "key_list: " + Style.RESET_ALL)
		print(key_list)

	return state_dict, reward_dict, action_dict, knob_list, key_list



def init_linux_mcd_dataset(df):
	reward_cols = ['joules_99', 'joules_per_interrupt', 'time_per_interrupt']
	skip_cols = ['fname', 'sys', 'core', 'exp', 'rapl']

	df_state = df.set_index(['itr', 'dvfs']).drop(reward_cols, axis=1).drop(skip_cols, axis = 1)
	df_reward = df.set_index(['itr', 'dvfs'])[reward_cols]
	state_dict = df_state.T.to_dict()
	for key in state_dict:
		state_dict[key] = np.array(list(state_dict[key].values()))
	reward_dict = df_reward.T.to_dict()
	action_dict, knob_list = prepare_action_dicts(df)
	key_list = list(state_dict.keys())

	if not debug:
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


def prepare_action_dicts(df, key_set):
#def prepare_action_dicts(df):
# NOTE: maybe skip dvfs's here..
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
			if pre == 1:
				pre = -1
			if suc == '0xffff':
				suc = -1
			d[elem] = {-1: pre, 0: elem, 1: suc}

			#d[elem] = {-1: l_m1[idx], 0: elem, 1: l_p1[idx]}
		return d

	d = {}
	knob_list = []
	for knob in ['itr', 'dvfs']:
		knob_list.append(knob)
		d[knob] = get_knob_dict(knob)	

	print(Fore.BLACK + Back.RED + "d[itr]:  " + str(d['itr']) + Style.RESET_ALL)
	print(Fore.BLACK + Back.RED + "d[dvfs]:  " + str(d['dvfs']) + Style.RESET_ALL)

	return d, knob_list






