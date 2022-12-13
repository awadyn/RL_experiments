import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os, sys, shutil, re

plt.ion()

debug = True

def process_logs(PATH):
	global SYSTEM

	data = []
	for f in os.listdir(PATH):
		#extract from filename
		run = re.search(r'linux\.mcd\.dmesg\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(1)
		core = re.search(r'linux\.mcd\.dmesg\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(2)
		itr = re.search(r'linux\.mcd\.dmesg\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(3)
		dvfs = re.search(r'linux\.mcd\.dmesg\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(4)
		qps = re.search(r'linux\.mcd\.dmesg\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(6)

		if debug:
			print("RUN, CORE, ITR-DELAY, DVFS, QPS, OS = ", run, core, itr, dvfs, qps, SYSTEM)

		features = process_file(PATH + '/' + f, itr, dvfs, qps, run, core)
		data.append(features)

	data = pd.DataFrame(data)
	print("Data: ")
	print(data)
	print()
	print()

	return data

def process_file(path, itr, dvfs, qps, run, core):
	global SYSTEM

	features = {}
	if SYSTEM == 'linux':
		#values = pd.read_csv(path, sep = ' ', index_col = 0, names = ['rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c1', 'c1e', 'c3', 'c6', 'c7', 'joules', 'timestamp'])
		values = pd.read_csv(path, sep = ' ', index_col = 0, names = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c1', 'c1e', 'c3', 'c6', 'c7', 'joules', 'timestamp'])
	elif SYSTEM == 'ebbrt':
		values = pd.read_csv(path, sep = ' ', index_col = 0, names = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c3', 'c6', 'c7', 'joules', 'timestamp'])

	if not debug:
		print(values.head())
		print(values.shape)
#		print(values['rx_bytes'].head())

	PCT_LIST = [10, 25, 50, 75, 90, 99]
	
	interrupt_cols = ['rx_bytes', 'rx_desc', 'tx_bytes', 'tx_desc']
	for c in interrupt_cols:
		pcts = np.percentile(values[c], PCT_LIST)
#		print(c, "percentiles: ")
#		print(pcts)
		for i, p in enumerate(PCT_LIST):
			features[f'{c}_{p}'] = pcts[i]
	    
	ms_cols = ['instructions', 'llc_miss', 'cycles', 'ref_cycles']
	for c in ms_cols:
		pcts = np.percentile(values[c], PCT_LIST)
		for i, p in enumerate(PCT_LIST):
			features[f'{c}_{p}'] = pcts[i]

	perf_cols = ['joules']
	for c in perf_cols:
		pct = np.percentile(values[c], 99)
		features[f'{c}_99'] = pct
		#print("joules: ", pct)

	#extract from filename
	features['itr'] = itr
	features['dvfs'] = dvfs
	features['qps'] = qps
	features['i'] = str(run) + '_' + str(core)

	features['time_per_interrupt'] = values['timestamp'].max() / values.shape[0]
	features['joules_per_interrupt'] = values['joules'].sum() / values.shape[0]
#	
#	features['lat_99'] = ...
	return features

if __name__ == '__main__':
	global SYSTEM

	log_data_dir = sys.argv[1]
	SYSTEM = sys.argv[2]
	featurized_logs_file = log_data_dir[0:len(log_data_dir)-1] + "_featurized.csv" 
	data = process_logs(log_data_dir)
	data.to_csv(featurized_logs_file, sep = ' ')

'''
if too slow, we'll spawn multiple threads

import multiprocessing as mp
manager = mp.Manager()
'''
