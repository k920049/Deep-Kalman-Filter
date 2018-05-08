import tensorflow as tf
import collections


_LSTMOutputTuple = collections.namedtuple("LSTMOutputTuple", ("z", "mu", "cov"))

class LSTMOutputTuple(_LSTMOutputTuple):

    __slots__ = ()

    @property
    def dtype(self):
        (z, mu, cov) = self

        if z.dtype == mu.dtype and mu.dtype == cov.dtype:
            return z.dtype
        else:
            raise TypeError("Inconsistent internal state: %s vs %s vs %s" % (str(z.dtype),
                                                                             str(mu.dtype),
                                                                             str(cov.dtype)))
