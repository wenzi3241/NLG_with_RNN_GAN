import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import helper
import numpy as np
import problem_unittests as tests
from collections import Counter
import tensorflow as tf
from tensorflow.contrib import seq2seq
import pickle

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    input = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return(input, targets, learning_rate)

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    #lstm_size = 256
    lstm_layers = 2
    
    # Basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = tf.identity(cell.zero_state(batch_size, tf.float32), name='initial_state')
    
    return(cell, initial_state)

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    #embedding_weight = helper.load_data("e")
    #print("vocab_size = ", vocab_size)
    #print("embed_dim = ", embed_dim)
    embedding_weight = pickle.load(open('embidding.p', mode='rb'))
    #tf.Variable(tf.truncated_normal((vocab_size, embed_dim), stddev = 0.01))
    embed_layer = tf.nn.embedding_lookup(embedding_weight, input_data)
    
    return embed_layer

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    
    return outputs, final_state

def build_nn(cell, rnn_size, input_data, vocab_size):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :return: Tuple (Logits, FinalState)
    """
    embed = get_embed(input_data, vocab_size, rnn_size)
    outputs, final_state = build_rnn(cell, embed)
    logits = tf.contrib.layers.fully_connected(outputs, 
                                               vocab_size, 
                                               weights_initializer = tf.truncated_normal_initializer(stddev = 0.01), 
                                               activation_fn=None)
    
    
    return logits, final_state

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    num_items = len(int_text)
    num_batches = num_items // (batch_size * seq_length)
    batches = np.zeros(shape = (num_batches, 2, batch_size, seq_length), dtype = np.int32)
    for item in range(num_batches):
        for batch in range(batch_size):
            start_x = batch * batch_size  * seq_length + item * seq_length
            end_x = start_x + seq_length
            if end_x < num_items:
                start_y = start_x + 1
                end_y = start_y + seq_length
                batches[item, 0, batch, :] = int_text[start_x:end_x]
                batches[item, 1, batch, :] = int_text[start_y:end_y]
        
    return batches

# Number of Epochs
num_epochs = 150
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 512
# Sequence Length
seq_length = 15
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 100

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gradients)

batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')

helper.save_params((seq_length, save_dir))
