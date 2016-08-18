from keras.models import load_model
from preprocess import char_indices, indices_char, chars, text
from sampling_utils.sample import sample
import numpy as np
import random
import os
from config import generating_config, training_config

config = generating_config

maxlen = training_config['maxlen']

model = load_model(os.path.join("models", config['model_name']))

# first we have to generate a seed sequence of length 40 (default)
if config['seed']:
    # use seed from config
    sentence = config['seed']
else:
    # generate sentence seed
    generated = ''
    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index: start_index + maxlen]
    generated += sentence

    # print seed
    print "----- seed ----- "
    print sentence
    print "----- generated -----"
# one-hot encode generated string in [1, 40, 60] matrix (put the text in a format the compute can understand)
for _ in xrange(config['length']):
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    #put our randomly pulled seed into our model
    #our model takes a sequence of 40 characters and predicts the next
    # input dimms (40, 60) output dims (60)
    preds = model.predict(x, verbose=0)[0]
    # sample from our 1/60 output
    next_index = sample(preds, config['diversity'])
    # convert one-hot vector back to string
    next_char = indices_char[next_index]
    generated += next_char
    sentence = sentence[1:] + next_char

if config['file']:
    with open(os.path.join('generated', config['file']), 'w') as outfile:
        outfile.write(generated)

print generated
