"""
"""
import tensorflow as tf
import numpy as np

from keras.layers import Layer, Conv2D


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class SpatioTemporalConv(Layer):
    """K-order Chebyshev graph convolution"""

    def __init__(self, spatial_filters, temporal_filters, cheb_polynomials, **kwargs):
        super(SpatioTemporalConv, self).__init__(**kwargs)
        self.spatial_filters = spatial_filters
        self.temporal_filters = temporal_filters
        self.cheb_polynomials = cheb_polynomials
        # weights and layers will be initialized in the "build" method
        self.cheb_weights = None
        self.temporal_convolution = None
        self.residual_convolution = None
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        """
        :param input_shape: (batch, times, links, features)
        :return:
        """
        # Chebyshev graph convolution params
        self.cheb_weights = self.add_weight(
            shape=(len(self.cheb_polynomials), input_shape[3], self.spatial_filters), initializer="glorot_uniform",
            trainable=True, name='cheb_weights', dtype=np.float32
        )
        # temporal convolution layer
        self.temporal_convolution = Conv2D(filters=self.temporal_filters, kernel_size=(3, 1), padding="same",
                                           activation='relu')
        # residual convolution
        self.residual_convolution = Conv2D(filters=self.temporal_filters, kernel_size=(1, 1), padding="same",
                                           activation='relu')

    def call(self, inputs, **kwargs):
        x = inputs

        # convolve
        x_graph_conv = self.chebyshev_convolution(x)
        x_temporal_conv = self.temporal_convolution(x_graph_conv)
        x_residual = self.residual_convolution(x)
        return tf.nn.relu(x_residual + x_temporal_conv)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], self.temporal_filters

    def chebyshev_convolution(self, x):
        (batch_size, num_of_timesteps, num_of_vertices, num_of_features) = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            x_time = x[:, time_step, :, :]
            x_time = tf.transpose(x_time, perm=(0, 2, 1))
            cheb_dot_results = list()
            for k in range(len(self.cheb_polynomials)):
                cheb_poly_k = self.cheb_polynomials[k]
                theta_k = self.cheb_weights[k]
                rhs = tf.transpose(dot(x_time, cheb_poly_k), perm=(0, 2, 1))
                cheb_dot_result = dot(rhs, theta_k)
                cheb_dot_results.append(cheb_dot_result)
            output = tf.add_n(cheb_dot_results)
            outputs.append(output)

        outputs = tf.stack(outputs, axis=1)
        return tf.nn.relu(outputs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'spatial_filters': self.spatial_filters,
            'temporal_filters': self.temporal_filters,
            'cheb_polynomials': self.cheb_polynomials
        })
        return config


class Conv2DDense(Layer):
    def __init__(self, kernel_size, output_dim, **kwargs):
        super(Conv2DDense, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.final_convolution = None

    def build(self, input_shape):
        self.final_convolution = Conv2D(filters=self.output_dim[1], kernel_size=self.kernel_size)
        super(Conv2DDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input: (batch, times, links, filters)
        # output: (batch, links, times)
        x_conv = self.final_convolution(inputs)
        output = x_conv[:, -1, :, :]
        return output

    def compute_output_shape(self, input_shape):
        return None, self.output_dim[0], self.output_dim[1]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'kernel_size': self.kernel_size
        })
        return config