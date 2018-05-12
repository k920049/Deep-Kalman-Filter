"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.array_ops import shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops.random_ops import random_normal
from tensorflow.python.ops.init_ops import random_normal_initializer
from tensorflow.python.ops.math_ops import sqrt

from src.LSTMOutputTuple import LSTMOutputTuple

class RNN(rnn_cell_impl.RNNCell):

    def __init__(self,
                 num_units,
                 reuse=None):
        # Override parent's initializer
        super(RNN, self).__init__(_reuse=reuse)

        self._num_units = num_units
        self._activation = math_ops.tanh
        self._reuse = reuse
        self._initializer = random_normal_initializer
        self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_units))
        self._output_size = (LSTMOutputTuple(num_units, num_units, num_units))

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def _linear(self,
                args,
                output_size,
                bias=True,
                bias_initializer=None,
                kernel_initializer=None):
        """Linear map: sum_i(args[i] * W[i]), where W[i] is a Variable.

        Args:
          args: a 2D Tensor or a list of 2D, batch x n, Tensors.
          output_size: int, second dimension of W[i].
          bias: boolean, whether to add a bias term or not.
          bias_initializer: starting value to initialize the bias
            (default is all zeros).
          kernel_initializer: starting value to initialize the weight.


        Returns:
          A 2D Tensor with shape [batch x output_size] taking value
          sum_i(args[i] * W[i]), where each W[i] is a newly created Variable.

        Raises:
          ValueError: if some of the arguments has unspecified or wrong shape.
        """
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError(
                    "linear is expecting 2D arguments: %s" % shapes)
            if shape[1].value is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                 "but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope) as outer_scope:
            weights = vs.get_variable("kernel",
                                      [total_arg_size, output_size],
                                      dtype=dtype,
                                      initializer=kernel_initializer)

            if len(args) == 1:
                res = math_ops.matmul(args[0], weights)
            else:
                res = math_ops.matmul(array_ops.concat(args, 1), weights)
            if not bias:
                return res

            with vs.variable_scope(outer_scope) as inner_scope:
                inner_scope.set_partitioner(None)

                if bias_initializer is None:
                    bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)

                biases = vs.get_variable("bias",
                                         [output_size],
                                         dtype=dtype,
                                         initializer=bias_initializer)
                res = nn_ops.bias_add(res, biases)

        return res

    def call(self, inputs, state):
        """
        A callable function
        :param inputs: [batch size][dimension]
        :param state: (cur_state, prev_state)
        :return:
        """
        softplus = nn_ops.softplus
        tanh = math_ops.tanh

        (cur_state, prev_state) = state

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        scope = vs.get_variable_scope()
        with vs.variable_scope(scope, initializer=self._initializer) as unit_scope:
            # Calculate the next state
            with vs.variable_scope("next_state") as next_scope:
                h_next = self._linear(args=prev_state, output_size=self._num_units)
                h_next = tanh(h_next)
                h_next = 0.5 * (cur_state + h_next)
            # Calculate the mean
            with vs.variable_scope("mean") as mean_scope:
                mu_t = self._linear(args=h_next, output_size=self._num_units)
            # Calculate the covariance
            with vs.variable_scope("covariance") as cov_scope:
                cov_t = self._linear(args=h_next, output_size=self._num_units)
                cov_t = softplus(cov_t) + 1e-6
            eps_t = random_normal(shape=[shape(inputs)[0], self._num_units])
            z_t = mu_t + sqrt(cov_t) * eps_t

        output = (LSTMOutputTuple(z_t, mu_t, cov_t))
        new_state = (rnn_cell_impl.LSTMStateTuple(h_next, cur_state))

        return output, new_state

    def get_scope_name(self):
        return vs.get_variable_scope()