import pandas as pd
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import plotly.graph_objects as go

import utils

np.set_printoptions(suppress = True)

# first, create dataframe for this ML experiment
df = pd.DataFrame()
featurized_logs_file = sys.argv[1]
df = pd.read_csv(featurized_logs_file, sep = ' ')
#df = pd.read_csv(featurized_logs_file, sep = ',')

# convert dvfs hex strings to integers
df['dvfs'] = df['dvfs'].apply(lambda x: int(x, 16)) 

# normalizing data
#for col in df.drop(['itr' , 'dvfs' , 'i' , 'qps'], axis = 1).columns:
for col in df.drop(['itr', 'i' , 'qps'], axis = 1).columns:
#for col in df.drop(['fname', 'sys' , 'core', 'exp'], axis = 1).columns:
	# sanity check
	if (df[col].max() - df[col].min() == 0):
		continue
	df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# next, create a supervised dataset from this dataframe
# 1 - create itr labels, create state vectors (removing core, exp, sys, app, rapl, qps, itr, and dvfs from state vector values)
# 2 - split dataset into train and test 

#df = df.drop(['fname'], axis=1).set_index(['core', 'sys', 'exp', 'rapl'])
df = df.set_index(['i'])
#df = df.set_index(['fname'])
print("df:  ", df)

#features = df.drop(['itr', 'dvfs', 'qps'], axis=1).values
features = df.drop(['itr', 'qps'], axis=1).values
#features = df.drop(['itr', 'dvfs', 'sys', 'core', 'exp'], axis=1).values
features = torch.from_numpy(features).float()

labels = df['itr'].values
#itrs = labels
#itrs.sort()
#itr_labels = {}
#l = 1
#for itr in set(itrs):
#	itr_labels[itr] = l
#	l+= 1
#for i in range(len(labels)):
#	itr = labels[i]
#	labels[i] = itr_labels[itr]
labels = torch.from_numpy(labels).float()


#test_features = features[1200:]
#test_labels = labels[1200:]
#features = features[0:1200]
#labels = labels[0:1200]
print("features:  " )
print(features.detach().numpy())
print()
#print("test_features:  " )
#print(test_features.detach().numpy())
#print()
print("labels:  ")
print(labels.detach().numpy())
print()
#print("test_labels:  " )
#print(test_labels.detach().numpy())


net = nn.Sequential(
#	nn.Linear(features.shape[1], 1)
	nn.Linear(features.shape[1], 2**8),
	nn.ReLU(), #rectified linear unit
	nn.Linear(2**8, 2**10),
	nn.ReLU(),
	nn.Linear(2**10, 2**12),
	nn.ReLU(),
#	nn.Linear(2**10, 2**8),
#	nn.ReLU(),
#	nn.Linear(2**8, 2**6),
#	nn.ReLU(),
	nn.Linear(2**12, 1),
#	nn.Softmax(dim=1)
)

#criterion = nn.BCELoss()
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

losses = []
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
		if i % freq == 0:
			print('epoch:', i, 'loss:', loss.detach().numpy())
		losses.append(loss.detach().numpy())	
	
		#step 3: backprop to update weights
		# compute gradients/derivatives - backprop
		# use gradients to update weights - gradient descent - w = w - 0.1 * deriv. loss w.r.t. w
		
		optimizer.zero_grad() #set previous buffers to zero
		loss.backward() #backprop
		optimizer.step() #update weights        
	    
	return net

N = 1000
net = train(N, net, features, labels, criterion, optimizer, debug=True)

fig = go.Figure(layout_title_text="log(loss), N=1000, lr=1e-3, NN=8_10_12")
fig.add_trace(go.Scatter(x=list(range(1, N)), y=np.log(losses), mode='markers'))
fig.show()

#torch.save(net, "NN_8_10_12_itr")

