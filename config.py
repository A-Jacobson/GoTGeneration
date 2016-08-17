
training_config = {
        "maxlen" : 40, # max length of sequences
        "step" : 3, # step size (will determine overlap of the sequences)
        "load_model" : True, # load pretrained
        "load_weights" : False, # if your training is cut off after 10 epoch but before training is finished, weights will have been saved in models/tmp/. , use this to recover them
        "model_name" : "LSTM.h5", # will add GRU
        "nb_epoch" : 1, # set to 1 if you want to train incrementally, set >=10 for full training
        "batch_size" : 256
}

generating_config = {
        "model_name" : "LSTM.h5",
        "seed" : False, # optionally, set your own seed here, must be 40 characters including spaces and punctuation
        "diversity" : 1.0,
        "length" : 1000,  # length of the generated text. one page ~= 28000 characters
        "file" : "fakegot.txt" # set to None to just print your text
}
