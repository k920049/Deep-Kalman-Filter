import tensorflow as tf
import numpy as np

from tensorflow.contrib.distributions import MultivariateNormalFullCovariance


class Network(object):

    def __init__(self, num_inputs, num_units, num_layers, num_levels, scope_r):
        # placeholder that holds inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.scope_r = scope_r
        self.init = tf.random_normal_initializer()

    def _build_recognition(self, input):
        # clearing stack
        stack = []
        # variable scope
        with tf.variable_scope(name_or_scope=self.scope_r, reuse=tf.AUTO_REUSE):
            # RNN-like Q network
            prev = None
            for l in range(self.num_levels):
                X = tf.reshape(tensor=input[l], shape=[1, self.num_inputs])
                cur = tf.layers.dense(inputs=X,
                                    units=self.num_units,
                                    kernel_initializer=self.init,
                                    name="dense_X")
                if prev is not None:
                    for i in range(self.num_layers):
                        Y = tf.layers.dense(inputs=prev,
                                            units=self.num_units,
                                            use_bias=False,
                                            kernel_initializer=self.init,
                                            name="dense_Y" + str(i))
                    cur = cur + Y
                # compute parameters
                mu = tf.layers.dense(inputs=cur,
                                     units=self.num_units,
                                     kernel_initializer=self.init,
                                     name="mu")
                logd = tf.layers.dense(inputs=cur,
                                       units=self.num_units,
                                       kernel_initializer=self.init,
                                       name="logd")
                u = tf.layers.dense(inputs=cur,
                                    units=self.num_units,
                                    kernel_initializer=self.init,
                                    name="u")
                cov = self.get_covariace(logd=logd[0], u=u)
                stack.append([mu, cov])

                prev = cur
        return stack

    # status -> [time step][features]
    # input -> [time step][1]
    def get_latent_samples(self, status):
        return self._build_recognition(input=status)

    def get_trainable(self):
        return tf.trainable_variables(scope=self.scope_r)

    @staticmethod
    def get_covariace(logd, u):
        # using the formula in the original paper
        D = tf.diag(tf.exp(logd))
        D_inverse = tf.matrix_inverse(D)
        nu = 1 / (tf.matmul(u, tf.matmul(D_inverse, tf.transpose(u))) + 1)
        rhs = tf.matmul(D_inverse, tf.matmul(
            tf.transpose(u), tf.matmul(u, D_inverse)))
        rhs = nu * rhs

        return D_inverse - rhs
