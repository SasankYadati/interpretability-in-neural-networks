'''
Use the trained LSTM model to analyse performance and interpretability.
Develop visualizations to summarize the same.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from lime import lime_text
from rnn_lstm_model import loadData, preprocessData

# load graph and restore variables
sess = tf.Session()
saver = tf.train.import_meta_graph('trained-models/rnn-lstm-model.meta')
saver.restore(sess, tf.train.latest_checkpoint('trained-models/'))

# load the placeholders
graph = tf.get_default_graph()
sess.run(tf.global_variables_initializer())

# load the the prediction tensor
prediction = graph.get_tensor_by_name('prediction')

# load and preprocess data to make predictions
imdb, train_x, train_y, valid_x, valid_y, test_x, test_y = loadData()
train_x, valid_x, test_x = preprocessData(train_x, valid_x, test_x)

print(train_x.shape)
print(train_y.shape)
print(train_x[0])

# print(graph.collections)
# print(sess.graph_def)
print(sess.run(prediction, {'input_seq_1:0':train_x, 'target_class:0':train_y}))

def predict(text):
    '''
    text is a list of strings of shape (NUM_SAMPLES)
    returns numpy array containing probablities of shape (NUM_SAMPLES,NUM_CLASSES).
    '''

    # preprocess text
    pass

    # 

    