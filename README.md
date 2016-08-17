# GoTGeneration

Project to generate new Game of Thrones book using LSTM neural networks using the existing 5 books as source material.

## Dependencies
This project was tested with Python 2.7.11, Keras, and Theano on windows 10. Training takes ~40 minutes per epoch with a GTX970, CUDA, CuDNNv5 and CNMEM.
Follow the instructions posted here to get a local deep learning environment working in windows:


1. http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/
2. http://ankivil.com/making-theano-faster-with-cudnn-and-cnmem-on-windows-10/


## The Data
I'm using the first 5 Game of thrones books concatenated into one large corpus. Unfortunately, I doubt I'm allowed to host the raw text (as unpleasant as it would be to read in that format) on github.
In spite of this, the books are out there. If you own them already, try googling Game of Thrones.txt. The model will work with any large corpus of text (>100k Characters), not just game of thrones books.

The versions of 4th and 5th books I found had ascii encoding issues, which is the reason for some of the pre-processing code.


## The Model
This is a Character level language model inspired by Andrej Kaparthy's cs231 lectures and The Unreasonable Effectiveness of Recurrent Neural Networks blog post (http://karpathy.github.io/2015/05/21/rnn-effectiveness/).


## Where to go from here
- Game of Thrones chapters start with the name of the character who's POV that chapter follows in all caps (for example, "JON", "ARYA").
generate a "JON" chapter by combining and training with only prior JON chapters.

- Train a model to speak like specific characters (Tyrion) by scraping only their dialogue from the books or books + show scripts

- generate sample hybrid GoT chapter from prior books and other sci-fantasy text e.g. lord of the rings

- replace proper nouns in other corpus with proper nouns in got and use new corpus to generate text (Joffrey as Ghollum?)
