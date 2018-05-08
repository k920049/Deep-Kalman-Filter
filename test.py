import tensorflow as tf
import numpy as np

from model.P_RNN import RNN
from model.Q_LSTM import LayerNormLSTMCell
from src.read_data import read_data

from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
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
    with tf.variable_scope("basic", reuse=tf.AUTO_REUSE):
        basic_cell = RNN(num_units=256, reuse=tf.AUTO_REUSE)
    LSTM_cell = LayerNormLSTMCell(num_units=256, reuse=tf.AUTO_REUSE)
    outputs, states = tf.nn.dynamic_rnn(cell=basic_cell, inputs=X, dtype=tf.float32)
    assert isinstance(outputs, LSTMOutputTuple), "Error"
    (z, mu, cov) = outputs

with tf.name_scope("miscellaneous"):
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for b in get_batch(X_train, 16):
        z = sess.run([z], feed_dict={X: b})
        print(z.shape)




