import numpy as np

def sample(a, diversity=1.0):
    # helper function to sample an index from a probability array
    # higher temperature means more sampling variability
    a = np.log(a) / diversity
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def generate_seed():
    pass
