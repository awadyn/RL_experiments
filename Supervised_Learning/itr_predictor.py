import pandas as pd
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import utils

#%matplotlib inline
#import matplotlib.pylab as plt

# first, create dataframe for this ML experiment
df = pd.DataFrame()
featurized_logs_file = sys.argv[1]
df = pd.read_csv(featurized_logs_file, sep = ',')
#df = utils.normalize(df)

# next, create a supervised dataset from this dataframe
# 1 - create itr labels, create state vectors (removing core, exp, sys, app, rapl, qps, itr, and dvfs from state vector values)
# 2 - split dataset into train and test 
df = df.drop(['fname'], axis=1).set_index(['core', 'sys', 'exp', 'rapl'])
print("df:  ", df)
features = df.drop(['itr'], axis=1).values
features = torch.from_numpy(features).float()
labels = df['itr'].values
labels = torch.from_numpy(labels).float()

test_features = features[1200:]
test_labels = labels[1200:]
features = features[0:1200]
labels = labels[0:1200]
print("features:  " )
print(features)
print()
print("test_features:  " )
print(test_features)
print()
print("labels:  ")
print(labels)
print()
print("test_labels:  " )
print(test_labels)


net = nn.Sequential(
#	nn.Linear(features.shape[1], 1)
	nn.Linear(features.shape[1], 2**8),
	nn.ReLU(), #rectified linear unit
	nn.Linear(2**8, 2**10),
	nn.ReLU(),
	nn.Linear(2**10, 2**12),
	nn.ReLU(),
	nn.Linear(2**12, 1),
	#nn.Sigmoid()
)

#criterion = nn.BCELoss()
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

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
		
		optimizer.zero_grad() #set previous buffers to zero
		loss.backward() #backprop
		optimizer.step() #update weights        
	    
	return net

net = train(1000, net, features, labels, criterion, optimizer, debug=True)

#torch.save(net, "nn_regression_0")

