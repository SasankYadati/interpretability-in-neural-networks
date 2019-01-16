'''
Build a LSTM model to perform sentiment analysis on IMDB dataset.
Save the model to the disk for further analysis.
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np

VOCAB_SIZE = 10000
PAD_VALUE = 0
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_SEQ_LEN = 256
WORD_VEC_DIMS = 50
LSTM_UNITS = 32

input_seq = tf.placeholder(tf.int32, [None, MAX_SEQ_LEN], name='input_seq')
target_class = tf.placeholder(tf.float32, [None, 1], name='target_class')

def loadData():
    imdb = keras.datasets.imdb

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

    train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
    test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

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
    
    train_data = np.reshape(train_data, (train_data.shape[0], MAX_SEQ_LEN))
    test_data = np.reshape(test_data, (test_data.shape[0], MAX_SEQ_LEN))

    return train_data, test_data

def buildModel():
    '''
    returns output, cost and optimizer as tensor ops.
    '''
    # embedding layer
    word_vec = tf.Variable(tf.truncated_normal([VOCAB_SIZE, WORD_VEC_DIMS]), dtype=tf.float32)
    input_vec = tf.nn.embedding_lookup(word_vec, input_seq)
    print(input_vec.shape)

    # rnn lstm layer
    rnn_cell = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)
    rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell, output_keep_prob=0.2)

    # finally, the rnn put together
    output, _ = tf.nn.dynamic_rnn(rnn_cell, input_vec, dtype=tf.float32)
    
    output = tf.layers.flatten(output)
    
    output = tf.layers.dense(output, 128)
    output = tf.nn.relu(output)
    
    output = tf.layers.dense(output, 1)
    output = tf.nn.sigmoid(output)
    
    cost = tf.losses.sigmoid_cross_entropy(target_class, output)
    tf.summary.scalar('Loss', cost)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # a list of metrics to measure accuracy, precision, recall, f1-score
    metrics = []

    accuracy = tf.metrics.accuracy(target_class, tf.round(output))
    tf.summary.scalar('Accuracy', accuracy[0])

    precision = tf.metrics.precision(target_class, tf.round(output))
    tf.summary.scalar('Precision', precision[0])

    recall = tf.metrics.recall(target_class, tf.round(output))
    tf.summary.scalar('Recall', recall[0])

    metrics.append(accuracy)
    metrics.append(precision)
    metrics.append(recall)

    return optimizer, cost, output, metrics

if __name__ == '__main__':
    imdb, train_x, train_y, test_x, test_y = loadData()
    train_x, test_x = preprocessData(train_x, test_x, 256)
    
    print(train_x.shape) # 25000 x 256
    print(train_y.shape) # 25000 x 1
    
    optimizer, cost, output, metrics = buildModel()
    
    num_batches = train_x.shape[0] // BATCH_SIZE
    
    initializer_g = tf.global_variables_initializer()
    initializer_l = tf.local_variables_initializer()
    
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir='LOGS/', graph=sess.graph)
        merged = tf.summary.merge_all()
        sess.run([initializer_g, initializer_l])
        
        for epoch in range(NUM_EPOCHS):
            print("Epoch {}".format(epoch))

            for batch in range(0, num_batches):
                l = batch*BATCH_SIZE
                r = min((batch+1)*BATCH_SIZE, train_x.shape[0]-1)
                
                batch_x = train_x[l:r]
                batch_y = train_y[l:r]

                _ = sess.run([optimizer], feed_dict={input_seq: batch_x, target_class: batch_y})
                
            # log summaries every epoch
            summary, _, _, _ = sess.run([merged, metrics[0], metrics[1], metrics[2]], {input_seq: train_x, target_class: train_y})
            writer.add_summary(summary, epoch)
            
        writer.close()