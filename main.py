import tensorflow as tf
import numpy as np

from tensorflow.python import debug as tf_debug

from model.Network import Network
from src.optimizer_helper import regularize, normalize, rescale
from src.read_data import read_data

save_path = "./resource/snapshot.ckpt"

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

def Main():

    X_train, y_train, X_valid, y_valid = read_data()

    X = tf.placeholder(dtype=tf.float32, shape=[None, X_train.shape[1], X_train.shape[2]])

    with tf.name_scope("network"):
        network = Network(num_transition_layers=3,
                          num_transition_units=512,
                          num_emission_layers=3,
                          num_emission_units=X_train.shape[2],
                          time_step=X_train.shape[1])

    with tf.name_scope("loss"):
        loss, nll, KL = network.neg_elbo(X)
        ops = []

        recognition_scope       = "recognition"
        generative_scope        = "generative"
        recognition_vars        = network.get_variable(recognition_scope)
        generative_vars         = network.get_variable(generative_scope)
        loss = regularize(cost=loss, params=recognition_vars, reg_val=0.05, reg_type='l2')
        loss = regularize(cost=loss, params=generative_vars, reg_val=0.05, reg_type='l2')


        recognition_optimizer   = tf.train.AdamOptimizer(learning_rate=1e-3)
        generative_optimizer    = tf.train.AdamOptimizer(learning_rate=1e-3)
        recognition_gv          = recognition_optimizer.compute_gradients(loss, recognition_vars)
        generative_gv           = generative_optimizer.compute_gradients(loss, generative_vars)
        recognition_grads       = [grad for grad, var in recognition_gv]
        generative_grads        = [grad for grad, var in generative_gv]
        recognition_vars        = [var for grad, var in recognition_gv]
        generative_vars         = [var for grad, var in generative_gv]
        recognition_grads       = normalize(recognition_grads, 1.0)
        generative_grads        = normalize(generative_grads, 1.0)
        recognition_gv          = list(zip(recognition_grads, recognition_vars))
        generative_gv           = list(zip(generative_grads, generative_vars))
        recognition_ops         = recognition_optimizer.apply_gradients(recognition_gv)
        generative_ops          = generative_optimizer.apply_gradients(generative_gv)

        ops.append(recognition_ops)
        ops.append(generative_ops)

    with tf.name_scope("miscellaneous"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        sess.run(init)
        epoch = 1
        index = 0

        batch_gen = get_batch(X_train, 128)

        while True:
            try:
                batch = next(batch_gen)
                batch = np.stack(batch, axis=0)
                index = index + 1
            except StopIteration:
                batch_gen = get_batch(X_train, 128)
                epoch = epoch + 1
                continue

            l, n, k, _ = sess.run([loss, nll, KL, ops], feed_dict={X: batch})
            print("At epoch {} iteration {}, loss -> {}, negative log likelihood -> {}, KL divergence -> {}".format(epoch, index, l, n, k))


if __name__ == "__main__":
    Main()