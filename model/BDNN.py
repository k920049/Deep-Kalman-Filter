import tensorflow as tf
import numpy as np

from model.Network import Network

class Q_BDNN(object):

    def __init__(self, state, input, num_layers, num_units, time_step, scope):

        self.state = state
        self.input = input
        self.num_layers = num_layers
        self.num_units = num_units
        self.time_step = time_step
        self.scope = scope

        assert (input.shape[0] == self.time_step), "Input dimension doesn't match with the time step"

        self.proposal

    def _forward_pass(self):
