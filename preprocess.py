from data_utils.clean import combine_documents, process_files
from data_utils.vectorize import make_sequences, vectorize, make_dicts
from config import training_config

config = training_config
maxlen = config['maxlen']
step = config['step']

chars, char_indices, indices_char, text = make_dicts()

if __name__ == "__main__":
    process_files()
    combine_documents()
    sentences, next_chars = make_sequences(maxlen=maxlen, step=step)
    vectorize(sentences, next_chars, chars, maxlen, char_indices)
