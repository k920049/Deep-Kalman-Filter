import tensorflow as tf
import numpy as np

from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

rand_init = tf.random_normal_initializer()
mu = tf.get_variable(name="mu", shape=[1, 16], dtype=tf.float32, initializer=rand_init)
log_d = tf.get_variable(name="log_d", shape=[1, 16], dtype=tf.float32, initializer=rand_init)

with tf.name_scope(name="sample"):
    cov = tf.diag(tf.exp(log_d[0]))
    dist = MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
    sample = dist.sample()
    prob = dist.prob(value=sample)

with tf.name_scope("gradient"):
    optimizer = tf.train.AdamOptimizer()
    list = optimizer.compute_gradients(loss=prob, var_list=[log_d])


with tf.name_scope("miscellaneous"):
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    res = sess.run(list)
    print(res)