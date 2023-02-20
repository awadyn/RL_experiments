import pandas as pd
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import re

import utils


debug = True


# first, create dataframe of featurized logs from runs with different QPS values
# input directory should have 1 file for each QPS value
featurized_logs_dir = sys.argv[1]
df = pd.DataFrame()
test_df = pd.DataFrame()

# create per-QPS dataframes
for qps_logs_file in os.listdir(featurized_logs_dir):
	filename = featurized_logs_dir + qps_logs_file
	print('filename: ' + filename)
	qps_df = pd.read_csv(filename, sep = ',')
	print(qps_df)
	itrs = []
	dvfss = []
	qpss = []
	runs = []
	cores = []	
	# adding itr, dvfs, and qps columns to df
	for i, v in qps_df.iterrows():
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
	qps_df['itr'] = itrs
	qps_df['dvfs'] = dvfss
	qps_df['qps'] = qpss
	qps_df['run'] = runs
	qps_df['core'] = cores

	total_size = len(qps_df)
	test_qps_df = qps_df.sample(n=int(total_size/4))
	# append per-QPS dataframe to full dataframe
	df = pd.concat([df, qps_df])
	test_df = pd.concat([test_df, test_qps_df])
	df = df.append(test_qps_df).drop_duplicates(keep=False)

energy_cols = ['joules_sum', 'joules_1', 'joules_10', 'joules_25', 'joules_50', 'joules_75', 'joules_90', 'joules_99', 'joules_per_interrupt']
id_cols = ['core', 'run']
knob_cols = ['itr', 'dvfs']
skip_cols = ['fname', 'qps']
reward_cols = ['core', 'run', 'joules_sum']
	
# normalize columns as needed
non_norm_cols = np.concatenate((energy_cols, id_cols, knob_cols, skip_cols), axis = 0) 
for col in df.drop(non_norm_cols, axis = 1).columns:
	# sanity check
	if ((df[col].max() - df[col].min() == 0) or (test_df[col].max() - test_df[col].min() == 0)):
		continue
	df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
	test_df[col] = (test_df[col] - test_df[col].min()) / (test_df[col].max() - test_df[col].min())

print()
print()
print('train dataframe:')
print(df)
print('test dataframe:')
print(df)
print()
print()

# next, create a supervised dataset from this dataframe
# 1 - create qps labels, create state vectors
# 2 - split dataset into train and test 
state_df = df.drop(non_norm_cols, axis=1)
test_state_df = test_df.drop(non_norm_cols, axis=1)
print()
print()
print("state_df:  ", state_df)
print("test_state_df:  ", test_state_df)
print()
print()

features = state_df.values
features = torch.from_numpy(features).float()
labels = df['qps'].values
labels = torch.from_numpy(labels).float()
test_features = test_state_df.values
test_features = torch.from_numpy(test_features).float()
test_labels = test_df['qps'].values
test_labels = torch.from_numpy(test_labels).float()
print("features:  " )
print(features)
print(features.shape)
print()
print("test_features:  " )
print(test_features)
print(test_features.shape)
print()
print("labels:  ")
print(labels)
print(labels.shape)
print()
print("test_labels:  " )
print(test_labels)
print(test_labels.shape)
print()

net = nn.Sequential(
	nn.Linear(features.shape[1], 2**8),
	nn.ReLU(), #rectified linear unit
	nn.Linear(2**8, 2**10),
	nn.ReLU(),
	nn.Linear(2**10, 2**12),
	nn.ReLU(),
	nn.Linear(2**12, 1),
#	nn.Linear(2**10, 5),
#	nn.Softmax(dim=1)
)

##criterion = nn.BCELoss()
##criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-2)

# validation/test function
def validate(net, test_features, test_labels):
	net_test_preds = net(test_features).squeeze()
	#np_preds = net_test_preds.detach().numpy()
	#np_preds = np_preds - (np.mod(np_preds, 5000))
	#net_test_preds = torch.from_numpy(np_preds).float()
	test_preds = torch.round(net_test_preds)

	if debug:
		print()
		print("test_preds:  ") 
		print(test_preds.detach().numpy())
		print("test_labels:  ") 
		print(test_labels.detach().numpy())
		
	test_accuracy = (np.abs(test_labels.detach().numpy() - test_preds.detach().numpy()) <= 50000).mean()
	test_rmse = np.sqrt((test_preds.detach().numpy() - test_labels.detach().numpy())**2).mean()
	return test_accuracy, test_rmse


# training function
def train(n_epochs, net, features, labels, criterion, optimizer, freq=10, debug=False):
	for i in range(n_epochs): #number of passes over full dataset
		# step 1: use net to make predictions
		net_preds = net(features).squeeze()
		
		# step 2: compute loss/error
		loss = criterion(net_preds, labels)

		# step 3: backprop to update weights
		optimizer.zero_grad() #set previous buffers to zero
		loss.backward() #backprop
		net.float()
		optimizer.step() #update weights        
	    
		if (debug and (i % freq == 0)):
			print()
			#print("net_preds:  ") 
			#print(net_preds.detach().numpy())
			#print("labels:  ") 
			#print(labels.detach().numpy())
			#print('epoch:', i, 'loss:', loss)
			train_accuracy, train_rmse = validate(net, features, labels)
			test_accuracy, test_rmse = validate(net, test_features, test_labels)
			net = net.train()
			print(f'Epoch: {i}    Loss: {loss}    Train Accuracy: {train_accuracy:.3f}    Test Accuracy: {test_accuracy:.3f}    Train RMSE: {train_rmse:.3f}    Test RMSE: {test_rmse:.3f}')

	return net

net = train(500, net, features, labels, criterion, optimizer, debug=True)






