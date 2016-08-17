import numpy as np
import random
import sys
import os
from models.lstm import create_lstm
from keras.callbacks import EarlyStopping

# load data into memory
path = os.path.join("data", "processed", "corpus.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

# indentify all uniuque characters then encode their positions in dicts
chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print char_indices

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print X[0].shape
# create model
model = create_lstm(input_shape=maxlen, len(chars))

# training settings
num_iters = 50 			#Number of iterations for training
epochs_per_iter = 25	#Number of iterations before we save our model
batch_size = 128		#Number of training examples pushed to the GPU per batch.
