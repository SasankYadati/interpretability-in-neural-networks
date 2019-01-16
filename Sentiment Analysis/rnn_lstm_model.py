'''
Build a LSTM model to perform sentiment analysis on IMDB dataset.
Save the model to the disk for further analysis.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import numpy as np


VOCAB_SIZE = 8000
PAD_VALUE = 0
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_SEQ_LEN = 256
WORD_VEC_DIMS = 70
LSTM_UNITS = 64
REGULARIZATION_CONSTANT = 0.0

input_seq = tf.placeholder(tf.int32, [None, MAX_SEQ_LEN], name='input_seq')
target_class = tf.placeholder(tf.float32, [None, 1], name='target_class')

def loadData():
    imdb = keras.datasets.imdb

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

    train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
    test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

    valid_data = train_data[0:5000]
    valid_labels = train_labels[0:5000]

    train_data = train_data[5000:]
    train_labels = train_labels[5000:]

    return imdb, train_data, train_labels, valid_data, valid_labels, test_data, test_labels

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

def preprocessData(train_data, valid_data, test_data):
    '''
    pad the arrays so they all have the same length,
    then create an integer tensor of shape max_length * num_reviews.
    we can use an embedding layer capable of handling this shape as the first layer in our network.
    '''

    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=PAD_VALUE, padding='post', maxlen=MAX_SEQ_LEN)
    valid_data = keras.preprocessing.sequence.pad_sequences(valid_data, value=PAD_VALUE, padding='post', maxlen=MAX_SEQ_LEN)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=PAD_VALUE, padding='post', maxlen=MAX_SEQ_LEN)
    
    train_data = np.reshape(train_data, (train_data.shape[0], MAX_SEQ_LEN))
    valid_data = np.reshape(valid_data, (valid_data.shape[0], MAX_SEQ_LEN))
    test_data = np.reshape(test_data, (test_data.shape[0], MAX_SEQ_LEN))

    return train_data, valid_data, test_data

def buildModel():
    '''
    returns output, cost and optimizer as tensor ops.
    '''
    # embedding layer
    word_vec = tf.Variable(tf.truncated_normal([VOCAB_SIZE, WORD_VEC_DIMS]), dtype=tf.float32, name='Word-Vectors')
    input_vec = tf.nn.embedding_lookup(word_vec, input_seq)
    print(input_vec.shape)

    # rnn lstm layer
    rnn_cell = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)
    rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell, output_keep_prob=0.25)

    # finally, the rnn put together
    output, _ = tf.nn.dynamic_rnn(rnn_cell, input_vec, dtype=tf.float32)
    
    output = tf.layers.flatten(output)

    output = tf.layers.dense(output, 128)
    output = tf.nn.relu(output)
    
    output = tf.layers.dense(output, 1)
    output = tf.nn.sigmoid(output)

    train_summaries = []
    valid_summaries = []

    loss = tf.losses.sigmoid_cross_entropy(target_class, output)
    train_summaries.append(tf.summary.scalar('Training-Loss', loss))
    valid_summaries.append(tf.summary.scalar('Validation-Loss', loss))

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # a list of metrics to measure accuracy, precision, recall, f1-score
    metrics = []

    output = tf.round(output)
    
    accuracy = tf.metrics.accuracy(target_class, output, name='Accuracy')
    train_summaries.append(tf.summary.scalar('Training-Accuracy', accuracy[0]))
    valid_summaries.append(tf.summary.scalar('Validation-Accuracy', accuracy[0]))

    precision = tf.metrics.precision(target_class, output, name='Precision')
    train_summaries.append(tf.summary.scalar('Training-Precision', precision[0]))
    valid_summaries.append(tf.summary.scalar('Validation-Precision', precision[0]))

    recall = tf.metrics.recall(target_class, output, name='Recall')
    train_summaries.append(tf.summary.scalar('Training-Recall', recall[0]))
    valid_summaries.append(tf.summary.scalar('Validation-Recall', recall[0]))

    metrics.append(accuracy)
    metrics.append(precision)
    metrics.append(recall)

    return optimizer, loss, output, metrics, train_summaries, valid_summaries

if __name__ == '__main__':
    imdb, train_x, train_y, valid_x, valid_y, test_x, test_y = loadData()
    train_x, valid_x, test_x = preprocessData(train_x, valid_x, test_x)

    print(train_x.shape) # 20000 x 256
    print(train_y.shape) # 20000 x 1
    
    optimizer, loss, output, metrics, train_summaries, valid_summaries = buildModel()
    
    num_batches = train_x.shape[0] // BATCH_SIZE
    
    initializer_g = tf.global_variables_initializer()
    initializer_l = tf.local_variables_initializer()
 
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        writer = tf.summary.FileWriter(logdir='LOGS/', graph=sess.graph)
        # merged = tf.summary.merge_all()
        
        sess.run([initializer_g, initializer_l])
        
        for epoch in range(NUM_EPOCHS):
            print("Epoch {}".format(epoch))

            for batch in range(0, num_batches):
                l = batch*BATCH_SIZE
                r = min((batch+1)*BATCH_SIZE, train_x.shape[0]-1)
                
                batch_x = train_x[l:r]
                batch_y = train_y[l:r]

                _, _, _, _ = sess.run([optimizer] + metrics, {input_seq: batch_x, target_class: batch_y})
                
            # log summaries every epoch
            training_summary, _, _, _ = sess.run([train_summaries]+metrics, {input_seq: train_x, target_class: train_y})
            validation_summary, _, _, _ = sess.run([valid_summaries]+metrics, {input_seq: valid_x, target_class: valid_y})

            for t_summ in training_summary:
                writer.add_summary(t_summ, epoch)
            for v_summ in validation_summary:
                writer.add_summary(v_summ, epoch)
            
        writer.close()