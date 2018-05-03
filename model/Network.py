import tensorflow as tf
import numpy as np

from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

class Network(object):

    def __init__(self, num_units, num_layers, num_levels, scope_r):
        # placeholder that holds inputs
        self.num_units = num_units
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.scope_r = scope_r
        self.default = tf.constant(value=[[1.0]], dtype=tf.float32)

    def _build_recognition(self, input):
        # clearing stack
        stack = []
        # variable scope
        with tf.variable_scope(name_or_scope=self.scope_r, reuse=tf.AUTO_REUSE):
            X = input
            for l in range(self.num_levels):
                with tf.variable_scope(name_or_scope="level" + str(l), reuse=tf.AUTO_REUSE)
                    for i in range(self.num_layers):
                        X = tf.layers.dense(inputs=X,
                                            units=self.num_units,
                                            name="level" + str(l) + "/dense/layer" + str(i))
                        X = tf.nn.elu(X)

                    # parameters
                    mu = tf.layers.dense(inputs=X,
                                         units=self.num_units,
                                         name="mu")
                    logd = tf.layers.dense(inputs=X,
                                           units=self.num_units,
                                           name="logd")
                    u = tf.layers.dense(inputs=X,
                                        units=self.num_units,
                                        name="u")
                    cov = self.get_covariace(logd, u)
                    stack.append([mu, cov])
        return stack
    # status -> [time step][features]
    # input -> [time step][1]
    def get_latent_samples(self, status, input):
        # concatenate status + input
        X = tf.concat([status, input], axis=1)
        return self._build_recognition(input=X)

    def get_trainable(self, scope):
        return tf.trainable_variables(scope=scope)

    @staticmethod
    def get_covariace(logd, u):
        # using the formula in the original paper
        D = tf.diag(tf.exp(logd))
        D_inverse = tf.matrix_inverse(D)
        nu = 1 / (tf.matmul(u, tf.matmul(D_inverse, tf.transpose(u))) + 1)
        rhs = tf.matmul(D_inverse, tf.matmul(tf.transpose(u), tf.matmul(u, D_inverse)))
        rhs = nu * rhs

        return D_inverse - rhs