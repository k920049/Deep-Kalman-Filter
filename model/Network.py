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
                 time_step):
        """
        Initializer for the model
        :param num_transition_layers:
        :param num_transition_units:
        :param num_emission_layers:
        :param num_emission_units:
        """
        self.num_transition_layers = num_transition_layers
        self.num_transition_units = num_transition_units
        self.num_emission_layers = num_emission_layers
        self.num_emission_units = num_emission_units
        self.time_step = time_step
        self.initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

        with tf.variable_scope(name_or_scope="recognition", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("LSTM"):
                self.recognition    = LayerNormLSTMCell(num_units=self.num_transition_units,
                                                        layer_norm=True,
                                                        initializer=self.initializer)
            with tf.variable_scope("aggregate"):
                self.aggregate      = RNN(num_units=self.num_transition_units)

    def _transition(self, z):
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
                                      kernel_initializer=self.initializer,
                                      bias_initializer=self.initializer,
                                      activation=tf.nn.tanh,
                                      name="dense" + str(l))
            mu = tf.layers.dense(inputs=hid,
                                 units=self.num_transition_units,
                                 kernel_initializer=self.initializer,
                                 bias_initializer=self.initializer,
                                 name="mu")
            cov = tf.layers.dense(inputs=hid,
                                  units=self.num_transition_units,
                                  kernel_initializer=self.initializer,
                                  bias_initializer=self.initializer,
                                  name="cov")
            cov = tf.nn.softplus(cov, name="cov_plus")
            cov = tf.clip_by_value(cov, clip_value_min=1e-3, clip_value_max=16.0)

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
                                      kernel_initializer=self.initializer,
                                      bias_initializer=self.initializer,
                                      activation=tf.nn.tanh,
                                      name="dense" + str(l))
            outp = tf.layers.dense(inputs=hid,
                                   units=self.num_emission_units * 2,
                                   kernel_initializer=self.initializer,
                                   bias_initializer=self.initializer,
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
        KL_mean = tf.reduce_mean(input_tensor=KL)

        return KL_sum, KL_mean

    def _q_z_x(self, X, anneal=1., ):
        """
        Prob(Z | X), where Z ~ Q(Z | X)
        :param X: [batch size][time step][dimension]
        :param dropout_prob:
        :param anneal:
        :return:
        """

        X = tf.reverse(X, axis=[1])
        outputs, states = tf.nn.dynamic_rnn(cell=self.recognition, inputs=X, dtype=tf.float32)
        assert isinstance(states, LSTMStateTuple), "Error: Type(states) is not LSTMStateTuple"
        outputs = tf.reverse(outputs, axis=[1])
        outputs = tf.layers.dropout(outputs, rate=0.1)

        z_q, mu_q, cov_q = self._aggregateLSTM(outputs)
        assert_cov = tf.assert_positive(cov_q)
        with tf.control_dependencies([assert_cov]):
            return z_q, mu_q, cov_q

    def _aggregateLSTM(self, hidden_state):
        """
        Compute parameters of Q distribution for later use
        :param hidden_state: [time step][batch size][dimension]
        :return: means and covariances
        """

        outputs, _ = tf.nn.dynamic_rnn(cell=self.aggregate, inputs=hidden_state, dtype=tf.float32)
        assert isinstance(outputs, LSTMOutputTuple), "Error: In aggregation network, the outputs calculated is not type \'LSTMOutputTuple\'"

        z = outputs.z
        mu = outputs.mu
        cov = outputs.cov

        assert_cov = tf.assert_positive(cov)
        with tf.control_dependencies([assert_cov]):
            return z, mu, cov

    def _nll_gaussian(self, mu, logcov, X):
        """
        Calculaate negative log likelihood
        :param mu:
        :param logcov:
        :param X:
        :param params:
        :return:
        """
        nll = 0.5 * (tf.log(2 * np.pi) + logcov + tf.divide(tf.square(X - mu), tf.exp(logcov)))
        return nll

    def neg_elbo(self, X, anneal=1.0):

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

            KL, KL_mean = self._temporalKL(mu_q, cov_q, mu_prior, cov_prior)
            hid_out = self._emission(z_q)

            mu_hid = tf.slice(hid_out, [0, 0, 0], [-1, -1, self.num_emission_units])
            logcov_hid = tf.slice(hid_out, [0, 0, self.num_emission_units], [-1, -1, self.num_emission_units])
            logcov_hid = tf.clip_by_value(logcov_hid, clip_value_min=-16.0, clip_value_max=16.0)
            nll_metric = self._nll_gaussian(mu_hid, logcov_hid, X)
            nll = tf.reduce_sum(nll_metric)
            # Evaluate negative ELBO
            neg_elbo = nll + anneal * KL

        return neg_elbo, nll, KL_mean

    def get_variable(self, scope):
        return tf.trainable_variables(scope=scope)

