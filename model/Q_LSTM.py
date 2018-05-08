"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

from src.layer import layer_norm


def _norm(g, b, inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(g)
    beta_init = init_ops.constant_initializer(b)
    with vs.variable_scope(scope):
        # Initialize beta and gamma for use by layer_norm.
        vs.get_variable("gamma", shape=shape, initializer=gamma_init)
        vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layer_norm(inp, reuse=True, scope=scope)
    return normalized


class LayerNormLSTMCell(rnn_cell_impl.RNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.

    The default non-peephole implementation is based on:

      http://www.bioinf.jku.at/publications/older/2604.pdf

    S. Hochreiter and J. Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

    The peephole implementation is based on:

      https://research.google.com/pubs/archive/43905.pdf

    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling." INTERSPEECH, 2014.

    The class uses optional peep-hole connections, optional cell clipping, and
    an optional projection layer.

    Layer normalization implementation is based on:

      https://arxiv.org/abs/1607.06450.

    "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

    and is applied before the internal nonlinearities.

    """

    def __init__(self,
                 num_units,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 num_proj=None,
                 proj_clip=None,
                 forget_bias=1.0,
                 activation=None,
                 layer_norm=False,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 reuse=None):
        """Initialize the parameters for an LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
            provided, then the projected values are clipped elementwise to within
            `[-proj_clip, proj_clip]`.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training. Must set it manually to `0.0` when restoring from
            CudnnLSTM trained checkpoints.
          activation: Activation function of the inner states.  Default: `tanh`.
          layer_norm: If `True`, layer normalization will be applied.
          norm_gain: float, The layer normalization gain initial value. If
            `layer_norm` has been set to `False`, this argument will be ignored.
          norm_shift: float, The layer normalization shift initial value. If
            `layer_norm` has been set to `False`, this argument will be ignored.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(LayerNormLSTMCell, self).__init__(_reuse=reuse)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh
        self._layer_norm = layer_norm
        self._norm_gain = norm_gain
        self._norm_shift = norm_shift

        if num_proj:
            self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_proj))
            self._output_size = num_proj
        else:
            self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_units))
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def _linear(self,
                args,
                output_size,
                bias,
                bias_initializer=None,
                kernel_initializer=None,
                layer_norm=False):
        """Linear map: sum_i(args[i] * W[i]), where W[i] is a Variable.

        Args:
          args: a 2D Tensor or a list of 2D, batch x n, Tensors.
          output_size: int, second dimension of W[i].
          bias: boolean, whether to add a bias term or not.
          bias_initializer: starting value to initialize the bias
            (default is all zeros).
          kernel_initializer: starting value to initialize the weight.
          layer_norm: boolean, whether to apply layer normalization.


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
            weights = vs.get_variable(
                "kernel", [total_arg_size, output_size],
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
                    bias_initializer = init_ops.constant_initializer(
                        0.0, dtype=dtype)
                biases = vs.get_variable(
                    "bias", [output_size], dtype=dtype, initializer=bias_initializer)

        if not layer_norm:
            res = nn_ops.bias_add(res, biases)

        return res

    def call(self, inputs, state):
        """Run one step of LSTM.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: this must be a tuple of state Tensors,
           both `2-D`, with column sizes `c_state` and
            `m_state`.

        Returns:
          A tuple containing:

          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        sigmoid = math_ops.sigmoid

        (c_prev, m_prev) = state

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]")
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope, initializer=self._initializer) as unit_scope:

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            lstm_matrix = self._linear([inputs, m_prev],
                                       4 * self._num_units,
                                       bias=True,
                                       bias_initializer=None,
                                       layer_norm=self._layer_norm)

            i, j, f, o = array_ops.split(value=lstm_matrix,
                                         num_or_size_splits=4,
                                         axis=1)

            if self._layer_norm:
                i = _norm(self._norm_gain, self._norm_shift, i, "input")
                j = _norm(self._norm_gain, self._norm_shift, j, "transform")
                f = _norm(self._norm_gain, self._norm_shift, f, "forget")
                o = _norm(self._norm_gain, self._norm_shift, o, "output")

            # Diagonal connections
            if self._use_peepholes:
                with vs.variable_scope(unit_scope):
                    w_f_diag = vs.get_variable(
                        "w_f_diag", shape=[self._num_units], dtype=dtype)
                    w_i_diag = vs.get_variable(
                        "w_i_diag", shape=[self._num_units], dtype=dtype)
                    w_o_diag = vs.get_variable(
                        "w_o_diag", shape=[self._num_units], dtype=dtype)

            if self._use_peepholes:
                c = (
                    sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
                    sigmoid(i + w_i_diag * c_prev) * self._activation(j))
            else:
                c = (
                    sigmoid(f + self._forget_bias) * c_prev +
                    sigmoid(i) * self._activation(j))

            if self._layer_norm:
                c = _norm(self._norm_gain, self._norm_shift, c, "state")

            if self._cell_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                c = clip_ops.clip_by_value(
                    c, -self._cell_clip, self._cell_clip)
                # pylint: enable=invalid-unary-operand-type
            if self._use_peepholes:
                m = sigmoid(o + w_o_diag * c) * self._activation(c)
            else:
                m = sigmoid(o) * self._activation(c)

            if self._num_proj is not None:
                with vs.variable_scope("projection"):
                    m = self._linear(m, self._num_proj, bias=False)

                if self._proj_clip is not None:
                    # pylint: disable=invalid-unary-operand-type
                    m = clip_ops.clip_by_value(
                        m, -self._proj_clip, self._proj_clip)
                    # pylint: enable=invalid-unary-operand-type

        new_state = (rnn_cell_impl.LSTMStateTuple(c, m))
        return m, new_state
