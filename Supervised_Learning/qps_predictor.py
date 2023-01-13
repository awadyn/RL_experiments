import pandas as pd
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import utils


# first, create dataframe for this ML experiment
featurized_logs_dir = sys.argv[1]
df = pd.DataFrame()
#test_df = pd.DataFrame()
for qps_logs in os.listdir(featurized_logs_dir):
	qps_df = pd.read_csv(featurized_logs_dir + qps_logs, sep = ' ')
	#total_size = len(qps_df)
	#test_qps_df = qps_df.sample(n=int(total_size/4))
	df = pd.concat([df, qps_df])
	#test_df = pd.concat([test_df, test_qps_df])
#df = utils.normalize(df)
#test_df = utils.normalize(test_df)

# next, create a supervised dataset from this dataframe
# 1 - create qps labels, create state vectors (removing core, exp, sys, app, itr, and dvfs from state vector values)
# 2 - split dataset into train and test 
df = df.drop(['itr','dvfs'], axis=1).set_index(['i'])
#test_df = test_df.drop(['itr','dvfs'], axis=1).set_index(['i'])
print("df:  ", df)
#print("test_df:  ", test_df)
features = df.drop(['qps'], axis=1)
labels = df['qps'].values
labels = torch.from_numpy(labels).float()
# normalize features
for col in features.columns:
	features[col] = (features[col] - features[col].mean())/features[col].std()
features = features.values
features = torch.from_numpy(features).float()

#test_features = test_df.drop(['qps'], axis=1).values
#test_features = torch.from_numpy(test_features).float()
#test_labels = test_df['qps'].values
#test_labels = torch.from_numpy(test_labels).float()

print("features:  " )
print(features)
print()
#print("test_features:  " )
#print(test_features)
#print()
print("labels:  ")
print(labels)
print()
#print("test_labels:  " )
#print(test_labels)

net = nn.Sequential(
	nn.Linear(features.shape[1], 2**8),
	nn.ReLU(), #rectified linear unit
#	nn.Linear(2**8, 2**10),
#	nn.ReLU(),
#	nn.Linear(2**10, 2**12),
#	nn.ReLU(),
	nn.Linear(2**8, 1),
#	nn.Sigmoid()
)

#criterion = nn.BCELoss()
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)


def train(n_epochs, net, features, labels, criterion, optimizer, freq=1, debug=False):
	for i in range(n_epochs): #number of passes over full dataset
		#step 1: get features and use net to make predictions
		net_preds = net(features).squeeze()
		preds = torch.round(net_preds)
		#preds = net(torch.from_numpy(features).float())
		#preds = np.asarray(preds.detach().numpy(), dtype='int')
		if debug: 
			print("preds:  ") 
			print(preds.detach().numpy())
		
		#step 2: compute loss/error
		#labels_torch = torch.tensor(torch.from_numpy(np.array(labels)).float())
		if debug: 
			print("labels:  ") 
			print(labels.detach().numpy())
		#loss = np.sqrt(((np.rint(preds.detach().numpy()) - labels_torch.detach().numpy())**2).mean())
		loss = criterion(net_preds, labels)
		if debug: 
			print("loss:  ") 
			print(loss.detach().numpy())
		if i % freq == 0:
			print('epoch:', i, 'loss:', loss)
		
		#step 3: backprop to update weights
		# compute gradients/derivatives - backprop
		# use gradients to update weights - gradient descent - w = w - 0.1 * deriv. loss w.r.t. w
#		
		optimizer.zero_grad() #set previous buffers to zero
		loss.backward() #backprop
		net.float()
		optimizer.step() #update weights        
	    
	return net

net = train(5, net, features, labels, criterion, optimizer, debug=True)






