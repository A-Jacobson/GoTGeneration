import numpy as np
import os
from model_utils.lstm import create_lstm
from keras.callbacks import ModelCheckpoint, EarlyStopping

# load training arrays
X = np.load(os.path.join("data", "train", "X.npy")) # of shape (num training examples(sequences), len(sequences), num unqiue chars)
y = np.load(os.path.join("data", "train", "y.npy"))

# create model
model = create_lstm(input_shape=(X.shape[1], X.shape[2]))

# training settings
nb_epoch = 10	        #Number of iterations before we save our model
batch_size = 128		#Number of training examples pushed to the GPU per batch.

# setup checkpoints to save new weights every epoch if those weights result in a lower loss
checkpointer = ModelCheckpoint(filepath=os.path.join("models", "tmp", "weights.hdf5"), verbose=1, save_best_only=True)
history = model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[checkpointer])

# save model architecure, weights, and loss hist
model.save("LSTM.h5")
cPickle.dump(history, open("training_hist/hist.p", "wb"))
