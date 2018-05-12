import tensorflow as tf
import numpy as np

from model.P_RNN import RNN
from model.Q_LSTM import LayerNormLSTMCell
# from src.read_data import read_data

from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.contrib.rnn import BasicRNNCell
from src.LSTMOutputTuple import LSTMOutputTuple

# X_train, y_train, X_valid, y_valid = read_data()
X_train = np.random.randn(500, 255, 500)

def get_batch(input, batch_size):

    batch = []

    for i in range(len(input)):
        # Append a single element
        batch.append(input[i])

        if len(batch) == batch_size:
            yield batch
            batch.clear()

    if len(batch) != 0:
        yield batch


X = tf.placeholder(dtype=tf.float32, shape=[None, X_train.shape[1], X_train.shape[2]])

with tf.name_scope("network"):
    with tf.variable_scope("cells", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("first"):
            basic_cell_layer_1 = BasicRNNCell(num_units=256)
            outputs, states = tf.nn.dynamic_rnn(basic_cell_layer_1, X, dtype=tf.float32)
        with tf.variable_scope("second"):
            basic_cell_layer_2 = LayerNormLSTMCell(num_units=256, layer_norm=True)
            outputs, states = tf.nn.dynamic_rnn(basic_cell_layer_2, outputs, dtype=tf.float32)
            assert isinstance(states, LSTMStateTuple), "Error"
            c = states.c
            h = states.h
            c = tf.reduce_mean(c)
            h = tf.reduce_mean(h)
            o = tf.reduce_mean(outputs)

    vars = tf.trainable_variables("cells")

with tf.name_scope("loss"):
    loss = c + h + o
    optimizer = tf.train.AdamOptimizer()
    grads_and_vars = optimizer.compute_gradients(loss=loss, var_list=vars)
    op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

with tf.name_scope("miscellaneous"):
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for b in get_batch(X_train, 16):
        _ = sess.run([op], feed_dict={X: b})




