import tensorflow as tf
import numpy as np

from keras.layers import Layer, Conv2D, LSTM
from keras.initializers import Zeros, Ones


class TransposeReshapeInput(Layer):
    def __init__(self, **kwargs):
        super(TransposeReshapeInput, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TransposeReshapeInput, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input: (batch, times, links, features)
        # output: (batch * links, times, features)
        input_shape = inputs.shape
        x = inputs
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        output = tf.reshape(x, shape=(-1, input_shape[1], input_shape[3]))
        return output

    def compute_output_shape(self, input_shape):
        return -1, input_shape[1], input_shape[3]


class ReshapeInput(Layer):
    def __init__(self, **kwargs):
        super(ReshapeInput, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReshapeInput, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input: (batch, times)
        # output: (batch, links, times)
        x = inputs
        output = tf.reshape(x, shape=(-1, inputs.shape[2]))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class ReshapeOutput(Layer):
    def __init__(self, output_shape, **kwargs):
        super(ReshapeOutput, self).__init__(**kwargs)
        self._output_shape = output_shape

    def build(self, input_shape):
        super(ReshapeOutput, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input: (batch, times)
        # output: (batch, links, times)
        x = inputs
        output = tf.reshape(x, shape=(-1, self._output_shape[0], self._output_shape[1]))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self._output_shape[0], self._output_shape[1]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            '_output_shape': self._output_shape
        })
        return config
