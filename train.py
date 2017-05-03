import numpy as np
import os
from keras.models import load_model
from model_utils.lstm import create_lstm
from keras.callbacks import ModelCheckpoint, EarlyStopping
from config import training_config

config = training_config

# load training arrays
X = np.load(os.path.join("data", "train", "X.npy")) #[:10000]  of shape (num training examples(sequences), len(sequences), num unqiue chars)
y = np.load(os.path.join("data", "train", "y.npy"))  #[:10000]
print X.shape
print y.shape

# create model
if config['load_model'] == True:
    model = load_model(os.path.join("models", config['model_name']))
else:
    model = create_lstm(input_shape=(X.shape[1], X.shape[2]))

if config['load_weights'] == True:
    model.load_weights(os.path.join('models', 'tmp', 'weights.hdf5'))

# setup checkpoints to save new weights every epoch if those weights result in a lower loss
earlystop = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath=os.path.join("models", "tmp", "weights.hdf5"), verbose=1, save_best_only=False, monitor='loss')
history = model.fit(X, y, batch_size=config['batch_size'], nb_epoch=config['nb_epoch'], verbose=1, callbacks=[checkpointer, earlystop])

# save model architecure, weights, and loss hist
model.save(os.path.join("models", "LSTM.h5"))

with open(os.path.join("training_hist", "hist.txt"), 'w') as outfile:
    outfile.write(str(history.history))

print history.history
