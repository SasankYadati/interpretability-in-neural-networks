'''
Build a LSTM model to perform sentiment analysis on IMDB dataset.
Save the model to the disk for further analysis.
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np

VOCAB_SIZE = 10000
EMBEDDING_SIZE = 10
PAD_VALUE = 0
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

def loadData():
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

    return imdb, train_data, train_labels, test_data, test_labels

def decodeExampleText(imdb, text):
    '''
    for given text, returns decoded form.
    numbers=>words
    '''
    word_index = imdb.get_word_index()
    # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()} 
    word_index["<PAD>"] = PAD_VALUE
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return ' '.join([reverse_word_index.get(i, '?') for i in text])

def preprocessData(train_data, test_data, max_length):
    '''
    pad the arrays so they all have the same length,
    then create an integer tensor of shape max_length * num_reviews.
    we can use an embedding layer capable of handling this shape as the first layer in our network.
    '''

    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=PAD_VALUE, padding='post', maxlen=max_length)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=PAD_VALUE, padding='post', maxlen=max_length)

    return train_data, test_data

def buildModel():
    # todo: formulate a lstm model to classify sequences of length 256
    pass

if __name__ == '__main__':
    imdb, train_x, train_y, test_x, test_y = loadData()
    train_x, test_x = preprocessData(train_x, test_x, 256)
    print(decodeExampleText(imdb, train_x[100]))