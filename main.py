import tensorflow as tf
import numpy as np

from model.Q_RNN import Q_RNN
from src.read_data import read_data

save_path = "./resource/snapshot.ckpt"

def Main():
    X_train, y_train, X_valid, y_valid = read_data()

    X = tf.placeholder(dtype=tf.float32, shape=[X_train.shape[1], X_train.shape[2]])

    with tf.name_scope("network"):
        network = Q_RNN(num_inputs=X_train.shape[2],
                        num_units=X_train.shape[2],
                        num_layers=3,
                        time_step=X_train.shape[1],
                        size=1,
                        scope="generative")
        proposal = network.proposal
        param_list = network.build_network(status=X)

    with tf.name_scope("loss"):
        ops = []
        loss = network.compute_loss(param_list=param_list)

        r_optimizer = tf.train.AdamOptimizer()
        g_optimizer = tf.train.AdamOptimizer()
        r_vars = proposal.get_trainable()
        g_vars = network.get_trainable()

        r_grad = r_optimizer.compute_gradients(loss=loss, var_list=r_vars)
        g_grad = g_optimizer.compute_gradients(loss=loss, var_list=g_vars)
        ops.append(r_optimizer.apply_gradients(grads_and_vars=r_grad))
        ops.append(g_optimizer.apply_gradients(grads_and_vars=g_grad))

    with tf.name_scope("miscellaneous"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        index = 0
        while True:
            index = index + 1
            e_idx = np.random.randint(low=0, high=X_train.shape[0] - 1)
            status = np.reshape(a=X_train[e_idx], newshape=[X_train.shape[1], X_train.shape[2]])
            l, _ = sess.run([loss, ops], feed_dict={X: status})
            print("At iteration {}, loss: {}".format(index, l))

            if index % 100 == 0:
                saver.save(sess=sess, save_path=save_path)

if __name__ == "__main__":
    Main()