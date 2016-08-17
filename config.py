# Preprocessing config
preprocessing_config = {
        "maxlen" : 40, # max length of sequences
        "step" : 3 # step size (will determine overlap of the sequences)
}

training_config = {
        "load_model" : True, # load pretrained
        "model_name" : "LSTM.h5", # will add GRU
        "nb_epoch" : 10, # set to 1 if you want to train incrementally, set >=10 for full training
        "batch_size" : 256
}

generating_config = {
        "model_name" : "LSTM.h5",
        "seed" : False, # optionally, set your own seed here, must be 40 characters including spaces and punctuation
        "diversity" : 1.,
        "length" : 400  # length of the generated text
}
