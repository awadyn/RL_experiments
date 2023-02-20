# this demo code will process a sequence of words and predict
# the next word from some index based on previous words/characters

import torch
from torch import nn
import numpy as np


# step 1: define input sequences (i.e. dataset)
text_corpus = ['hey how are you', 'good i am fine', 'have a nice day']
# NOTE: for our controller, this will be a list of log sequences
#	logs = [list of log files]
#	     = [[file = list of log entries], ...]
#	     = [[[entry = list of measurements], ...], ...]
#	     = [[[m_0, m_1, ...], [m_0, m_1, ...], ...], ...]


# step 2: create encoding scheme for text input sequences
alphabet = set(''.join(text_corpus))		# set of characters in input dataset
i2c = dict(enumerate(alphabet))		# dictionary of int:char mappings
# c2i: returns dictionary of char:int mappings from int:char mapping rules
c2i = {c:i for i,c in i2c.items()}	
# NOTE: for our controller, there will be no encoding needed


# step 3: pad inputs into equal lengths
maxlen = len(max(text_corpus, key=len))
for i in range(len(text_corpus)):
	while len(text_corpus[i]) < maxlen:
		text_corpus[i] += ' '
# NOTE: for our controller, there will be no padding needed
# NOTE: if porting to C, do not do this here; pad inputs upon creation


# step 4: prepare inputs and targets (i.e. items to predict)
#	  -> an input to RNN is useful except for its last character 
#	  -> a target can be any input character except for the first
input_corpus = []
target_corpus = []
for i in range(len(text_corpus)):
	input_corpus.append(text_corpus[i][:-1])
	target_corpus.append(text_corpus[i][1:])
	print(f'Input Sequence: {input_corpus[i]} - Target Sequence: {target_corpus[i]}')
# NOTE: for our controller
#	-> an input log file is useful except for its last entry
#	-> a target can be any log entry except for the first


# step 5: convert text input sequences to numeric sequences
for i in range(len(text_corpus)):
	input_corpus[i] = [c2i[c] for c in input_corpus[i]]
	target_corpus[i] = [c2i[c] for c in target_corpus[i]]
# NOTE: for our controller, this conversion is not needed


corpus_size = len(text_corpus)
alphabet_size = len(alphabet) 
seq_len = maxlen - 1
# step 6: convert text input sequences to one-hot-encodings
#	  these will be the final features to use as input to RNN
def one_hot_encode(corpus, alphabet_size, seq_len, corpus_size):
	features = np.zeros((corpus_size, seq_len, alphabet_size), dtype=np.float32)
	for i in range(corpus_size):
		for c in range(seq_len):
			features[i][c][corpus[i][c]] = 1
	return features
input_corpus = one_hot_encode(input_corpus, alphabet_size, seq_len, corpus_size)
# NOTE: for our controller, this conversion is not needed


# step 7: convert inputs and targets to torch tensors
input_corpus = torch.from_numpy(input_corpus)
target_corpus = torch.Tensor(target_corpus)


# step 8: define RNN model
class Model(nn.Module):
	def __init__(self, input_size, output_size, hidden_dim, n_layers):
		super(Model, self).__init__()
		# number of hidden nodes
		self.hidden_dim = hidden_dim
		# number of hidden layers
		self.n_layers = n_layers
		self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
		# linear fully-connected layer after rnn cells
		self.fc = nn.Linear(hidden_dim, output_size) 

	def forward(self, x):
		# x represents one batch of inputs to pass through rnn
		batch_size = x.size(0)
		# initialize hidden layers to zeros
		hidden = self.init_hidden(batch_size)
		# out: mapping of every character to a vector of size hidden_dim
		# hidden: updated hidden layer from one forward pass of rnn
		out, hidden = self.rnn(x, hidden)
		out = out.contiguous().view(-1, self.hidden_dim)
		# out: updated output after pass through linear layer
		out = self.fc(out)
		return out, hidden

	def init_hidden(self, batch_size):
		hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
		return hidden
model = Model(input_size=alphabet_size, output_size=alphabet_size, hidden_dim=12, n_layers=1)
n_epochs = 1000
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# step 9: train model
for epoch in range(1, n_epochs+1):
	optimizer.zero_grad()		# initialization
	output, hidden = model(input_corpus)
	loss = criterion(output, target_corpus.view(-1).long())
	loss.backward()
	optimizer.step()

	if epoch%5 == 0:
		print(f'Epoch: {epoch}  -  Loss: {loss.item():.4f}')


# step 10: test model
def predict(model, characters):
	# one-hot-encode single text input to 3-dimensional corpus of size
	#	= # text inputs  x  size of text input  x  size of alphabet
	characters = np.array([[c2i[c] for c in characters]])
	characters = one_hot_encode(characters, alphabet_size, characters.shape[1], 1)
	characters = torch.from_numpy(characters)
	# pass encoded input through model
	out, hidden = model(characters)
	# prob distribution representing next letter likelihood
	prob = nn.functional.softmax(out[-1], dim=0).data
	char_index = torch.max(prob, dim=0)[1].item()
	return i2c[char_index], hidden

def sample(model, out_len, start='hey'):
	model.eval()
	start = start.lower()
	chars = [c for c in start]
	size = out_len - len(chars)
	for i in range(size):
		char, h = predict(model, chars)
		chars.append(char)
	return ''.join(chars)


# test
print(sample(model, 15, 'good'))






