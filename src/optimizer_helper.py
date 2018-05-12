import tensorflow as tf


def regularize(cost, params, reg_val, reg_type):
    """
    Return a theano cost
    :param cost: cost to regularize
    :param params: list of parameters
    :param reg_val: multiplier for regularizer
    :param reg_type: accepted types of regularizer(options: 'l1' or 'l2'
    :param reg_spec:
    :return:
    """

    l1 = lambda p: tf.reduce_sum(tf.abs(p))
    l2 = lambda p: tf.reduce_sum(tf.square(p))
    rFxn = {}
    rFxn['l1'] = l1
    rFxn['l2'] = l2

    if reg_type == 'l2' or reg_type == 'l1':
        assert reg_val is not None, 'Expecting reg_val to be specified'
        regularizer = 0.0
        for p in params:
            regularizer = regularizer + rFxn[reg_type](p)
        return cost + reg_val * regularizer
    else:
        return cost


def normalize(grads, grad_norm):
    """
    Normalize the gradients
    :param grads: list of gradients
    :param grad_norm: None
    :return: gradients rescaled to satisfy norm constraints
    """

    # Check if we're clipping gradients
    if grad_norm is not None:
        assert grad_norm > 0, 'Must specify a positive value to normalize to'
        g2 = 0.0
        for g in grads:
            g2 = g2 + tf.reduce_sum(tf.square(g))
        new_grads = []
        for g in grads:
            new_grads.append(tf.cond(pred=(g2 > (tf.square(grad_norm))),
                                     true_fn=lambda: tf.divide(g, tf.sqrt(g2)),
                                     false_fn=lambda: tf.identity(g)))
        return new_grads
    else:
        return grads


def rescale(grads, divide_grad):
    """
    Rescaled gradients
    :param grads: list of gradients
    :param divide_grad: scalar
    :return: gradients divided by provided variable
    """

    if divide_grad is not None:
        new_grads = []

        for g in grads:
            new_grads.append(tf.divide(g, divide_grad))
        return new_grads
    else:
        return grads
