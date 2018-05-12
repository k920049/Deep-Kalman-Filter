import tensorflow as tf
import numpy as np

from model.Q_LSTM import LayerNormLSTMCell
from model.P_RNN import RNN

from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from src.LSTMOutputTuple import LSTMOutputTuple

class Network(object):

    def __init__(self,
                 num_transition_layers,
                 num_transition_units,
                 num_emission_layers,
                 num_emission_units,
                 time_step,
                 scope):
        """
        Initializer for the model
        :param num_transition_layers:
        :param num_transition_units:
        :param num_emission_layers:
        :param num_emission_units:
        :param scope:
        """
        self.num_transition_layers = num_transition_layers
        self.num_transition_units = num_transition_units
        self.num_emission_layers = num_emission_layers
        self.num_emission_units = num_emission_units
        self.time_step = time_step
        self.scope = scope

        with tf.variable_scope(name_or_scope=self.scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(name_or_scope="recognition"):
                # self.recognition    = LayerNormLSTMCell(num_units=self.num_transition_units, layer_norm=True)
                self.aggregate      = RNN(num_units=self.num_transition_units)

    def _transition(self, z, eps=1e-3):
        """
        Transition function that generates next hidden units
        :param z: [batch size][time step][dimension], previous hidden states
        :param trans_params:
        :return: mean and covariance of the next hidden states
        """
        with tf.variable_scope(name_or_scope="transition", reuse=tf.AUTO_REUSE):
            hid = z

            for l in range(self.num_transition_layers):
                hid = tf.layers.dense(inputs=hid,
                                      units=self.num_transition_units,
                                      kernel_initializer=tf.random_normal_initializer,
                                      activation=tf.nn.tanh,
                                      name="dense" + str(l))
            mu = tf.layers.dense(inputs=hid,
                                 units=self.num_transition_units,
                                 kernel_initializer=tf.random_normal_initializer,
                                 name="mu")
            cov = tf.layers.dense(inputs=hid,
                                  units=self.num_transition_units,
                                  kernel_initializer=tf.random_normal_initializer,
                                  name="cov")
            cov = tf.nn.softplus(cov, name="cov_plus") + 1e-6

            assert_cov = tf.assert_positive(cov)

            with tf.control_dependencies([assert_cov]):
                return mu, cov

    def _emission(self, z):
        """
        Emission function that generates observed variable space
        :param z: current latent state
        :return: Prob(X | Z)
        """
        with tf.variable_scope(name_or_scope="emission", reuse=tf.AUTO_REUSE):
            hid = z

            for l in range(self.num_emission_layers):
                hid = tf.layers.dense(inputs=hid,
                                      units=self.num_transition_units,
                                      kernel_initializer=tf.random_normal_initializer,
                                      activation=tf.nn.tanh,
                                      name="dense" + str(l))
            outp = tf.layers.dense(inputs=hid,
                                   units=self.num_emission_units * 2,
                                   kernel_initializer=tf.random_normal_initializer,
                                   name="outp")
        return outp

    def _temporalKL(self, mu_q, cov_q, mu_prior, cov_prior):
        """
        A function that calculates KL divergence in robust way
        :param mu_q: [batch size][time step][dimension]
        :param cov_q:
        :param mu_prior:
        :param cov_prior:
        :return:
        """
        diff_mu = mu_prior - mu_q
        KL = tf.log(cov_prior) - tf.log(cov_q) - 1.0 + cov_q / cov_prior + diff_mu ** 2 / cov_prior
        KL = 0.5 * tf.reduce_sum(input_tensor=KL, axis=2)
        KL_sum = tf.reduce_sum(input_tensor=KL)

        return KL_sum

    def _q_z_x(self, X, anneal=1., ):
        """
        Prob(Z | X), where Z ~ Q(Z | X)
        :param X: [batch size][time step][dimension]
        :param dropout_prob:
        :param anneal:
        :return:
        """

        outputs, states = tf.nn.dynamic_rnn(cell=self.aggregate, inputs=X, dtype=tf.float32)
        assert isinstance(outputs, LSTMOutputTuple), "Error: The output is not LSTMOutputTuple"

        z_q     = outputs.z
        mu_q    = outputs.mu
        cov_q   = outputs.cov

        z_q     = tf.reverse(z_q, axis=[1])
        mu_q    = tf.reverse(mu_q, axis=[1])
        cov_q   = tf.reverse(cov_q, axis=[1])

        assert_cov = tf.assert_positive(cov_q)

        with tf.control_dependencies([assert_cov]):
            return z_q, mu_q, cov_q
    """
    def _aggregateLSTM(self, hidden_state):
        
        Compute parameters of Q distribution for later use
        :param hidden_state: [time step][batch size][dimension]
        :return: means and covariances
        
        outputs, _ = tf.nn.dynamic_rnn(cell=self.aggregate, inputs=hidden_state, dtype=tf.float32)
        assert isinstance(outputs, LSTMOutputTuple), "Error: In aggregation network, the outputs calculated is not type \'LSTMOutputTuple\'"

        z = outputs.z
        mu = outputs.mu
        cov = outputs.cov

        assert_cov = tf.assert_positive(cov)
        with tf.control_dependencies([assert_cov]):
            return z, mu, cov
    """
    def _nll_gaussian(self, mu, logcov, X, params = None):
        """
        Calculaate negative log likelihood
        :param mu:
        :param logcov:
        :param X:
        :param params:
        :return:
        """
        nll = 0.5 * (tf.log(2 * np.pi) + logcov + tf.divide(tf.square(X - mu), tf.exp(logcov)))
        nll = tf.clip_by_value(nll, 1e-10, 1.0)
        return nll

    def neg_elbo(self, X, anneal=1.0):

        with tf.variable_scope(name_or_scope=self.scope, reuse=tf.AUTO_REUSE):

            with tf.variable_scope(name_or_scope="recognition"):
                z_q, mu_q, cov_q = self._q_z_x(X, anneal=anneal)

            with tf.variable_scope(name_or_scope="generative"):
                mu_trans, cov_trans = self._transition(z_q)
                mu_shape = tf.shape(mu_trans)
                cov_shape = tf.shape(cov_trans)
                mean_init = tf.zeros(shape=[mu_shape[0], 1, mu_shape[2]], dtype=tf.float32)
                cov_init = tf.ones(shape=[cov_shape[0], 1, cov_shape[2]], dtype=tf.float32)
                mu_prior = tf.concat([mean_init, mu_trans[:, :-1, :]], axis=1)
                cov_prior = tf.concat([cov_init, cov_trans[:, :-1, :]], axis=1)

                KL = self._temporalKL(mu_q, cov_q, mu_prior, cov_prior)
                hid_out = self._emission(z_q)

                mu_hid = tf.slice(hid_out, [0, 0, 0], [-1, -1, self.num_emission_units])
                logcov_hid = tf.slice(hid_out, [0, 0, self.num_emission_units], [-1, -1, self.num_emission_units])
                nll_metric = self._nll_gaussian(mu_hid, logcov_hid, X)
                nll = tf.reduce_sum(nll_metric)
                # Evaluate negative ELBO
                neg_elbo = nll + anneal * KL


        return neg_elbo

    def get_variable(self, scope):
        return tf.trainable_variables(scope=scope)

