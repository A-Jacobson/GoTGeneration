from data_utils.clean import combine_documents, process_files
from data_utils.vectorize import make_sequences, vectorize, make_dicts
from config import preprocessing_config

config = preprocessing_config()
maxlen = config['MAXLEN']
step = config['STEP']

process_files()
combine_documents()

chars, char_indices, indices_char = make_dicts()
sentences, next_chars = make_sequences(maxlen=maxlen, step=step)
vectorize(sentences, next_chars, chars, maxlen, char_indices)
