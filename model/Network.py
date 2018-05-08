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
                self.recognition    = LayerNormLSTMCell(num_units=self.num_transition_units, layer_norm=True)
                self.aggregate      = RNN(num_units=self.num_transition_units)

    def _transition(self, z, trans_params=None):
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
                                  activation=tf.nn.softplus,
                                  name="cov")
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
        assert_cov = tf.assert_positive(x=cov_q)
        assert_cov_prior = tf.assert_positive(x=cov_prior)

        with tf.control_dependencies(control_inputs=[assert_cov, assert_cov_prior]):
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
        embedding = tf.layers.dense(inputs=X,
                                    units=self.num_transition_units,
                                    kernel_initializer=tf.random_normal_initializer,
                                    activation=tf.nn.tanh,
                                    name="embedding")
        print(embedding.shape)
        outputs, states = tf.nn.dynamic_rnn(cell=self.recognition, inputs=embedding, dtype=tf.float32)
        assert isinstance(states, LSTMStateTuple), \
            "Error: In recognition network, the states calculated is not type \'LSTMStateTuple\'"

        states = outputs
        states = tf.reverse(states, [1])

        z_q, mu_q, cov_q = self._aggregateLSTM(hidden_state=states)

        return z_q, mu_q, cov_q

    def _aggregateLSTM(self, hidden_state):
        """
        Compute parameters of Q distribution for later use
        :param hidden_state: [time step][batch size][dimension]
        :return: means and covariances
        """
        outputs, _ = tf.nn.dynamic_rnn(cell=self.aggregate, inputs=hidden_state, dtype=tf.float32)
        assert isinstance(outputs, LSTMOutputTuple), \
            "Error: In aggregation network, the outputs calculated is not type \'LSTMOutputTuple\'"

        (z, mu, cov) = outputs
        return z, mu, cov

    def _nll_gaussian(self, mu, logcov, X, params = None):
        """
        Calculaate negative log likelihood
        :param mu:
        :param logcov:
        :param X:
        :param params:
        :return:
        """
        nll = 0.5 * (np.log(2 * np.pi) + logcov + (X - mu) ** 2 / tf.exp(logcov))
        return nll

    def neg_elbo(self, X, anneal=1.0):

        with tf.variable_scope(name_or_scope=self.scope, reuse=tf.AUTO_REUSE):

            with tf.variable_scope(name_or_scope="recognition"):
                z_q, mu_q, cov_q = self._q_z_x(X, anneal=anneal)

            with tf.variable_scope(name_or_scope="generative"):
                mu_trans, cov_trans = self._transition(z_q)
                mu_prior = tf.concat([tf.zeros_like(mu_trans[:, 0, :]), mu_trans[:, :-1, :]], axis=1)
                cov_prior = tf.concat([tf.ones_like(mu_trans[:, 0, :]), cov_trans[:, :-1, :]], axis=1)

                KL = self._temporalKL(mu_q, cov_q, mu_prior, cov_prior)
                hid_out = self._emission(z_q)

                mu_hid = tf.slice(hid_out, [0, 0, 0], [-1, -1, self.num_emission_units])
                logcov_hid = tf.slice(hid_out, [0, 0, self.num_emission_units], [-1, -1, 2 * self.num_emission_units])
                nll_metric = self._nll_gaussian(mu_hid, logcov_hid, X)
                nll = tf.reduce_sum(nll_metric)
                # Evaluate negative ELBO
                neg_elbo = nll + anneal * KL


        return neg_elbo

    def get_variable(self, scope):
        return tf.trainable_variables(scope=scope)

