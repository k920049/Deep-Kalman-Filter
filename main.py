import tensorflow as tf
import numpy as np

from model.Network import Network

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

    # X_train, y_train, X_valid, y_valid = read_data()
    dummy = np.random.randn(500, 250, 500)

    X = tf.placeholder(dtype=tf.float32, shape=[None, dummy.shape[1], dummy.shape[2]])

    with tf.name_scope("network"):
        network = Network(num_transition_layers=3,
                          num_transition_units=512,
                          num_emission_layers=3,
                          num_emission_units=dummy.shape[2],
                          time_step=dummy.shape[1],
                          scope="network")

    with tf.name_scope("loss"):
        loss = network.neg_elbo(X)
        ops = []

        recognition_scope       = "network/recognition"
        generative_scope        = "network/generative"
        recognition_vars        = network.get_variable(recognition_scope)
        generative_vars         = network.get_variable(generative_scope)
        recognition_optimizer   = tf.train.AdamOptimizer()
        generative_optimizer    = tf.train.AdamOptimizer()
        recognition_grad        = recognition_optimizer.compute_gradients(loss, recognition_vars)
        generative_grad         = generative_optimizer.compute_gradients(loss, generative_vars)
        recognition_ops         = recognition_optimizer.apply_gradients(recognition_grad)
        generative_ops          = generative_optimizer.apply_gradients(generative_grad)

        ops.append(recognition_ops)
        ops.append(generative_ops)

    with tf.name_scope("miscellaneous"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        epoch = 1
        index = 0

        batch_gen = get_batch(dummy)

        while True:
            index = index + 1
            try:
                batch = next(batch_gen)
            except StopIteration:
                break

            loss, _ = sess.run([loss, ops], feed_dict={X: batch})
            print("At epoch {} iteration {}, loss -> {}".format(epoch, index, loss))


if __name__ == "__main__":
    Main()