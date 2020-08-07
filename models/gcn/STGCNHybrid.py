"""
Tensorflow model (not Sequential)
"""
import time

from keras.models import Model
from keras.layers import Dropout, Dense, Reshape, Permute
from keras.layers import Input, add, concatenate
from models.base_model import *
from models.gcn.layers import *
from models.recurrent.layers import *

from models.gcn.utils.graph_utils import *


class STGCNHybridModel(BaseModel):
    def __init__(self, config):
        """
        Initialize the model
        """
        BaseModel.__init__(self, config)

        # common settings
        data_config = self._config['Data']
        adj_filename = data_config['adj_filename']
        adj_file_type = data_config['adj_file_type']
        points_per_hour = int(self._config['Data']['points_per_hour'])
        num_for_predict = int(self._config['Data']['num_for_predict'])
        num_of_vertices = int(data_config['num_of_vertices'])

        training_config = self._config['Training']
        num_of_weeks = int(training_config['num_of_weeks'])
        num_of_days = int(training_config['num_of_days'])
        num_of_hours = int(training_config['num_of_hours'])

        model_config = self._config['Model']
        max_chebyshev_poly_order = int(model_config['K'])

        feature_vector_len = num_of_hours * points_per_hour
        feature_vectors_count = num_of_weeks * 2 + num_of_days * 2 + 1
        spatial_filters = 64
        temporal_filters = 64
        lstm_units = 64

        adjacency_matrix = get_adjacency_matrix(adj_filename, num_of_vertices, file_type=adj_file_type, scaling=False)
        cheb_polynomials = chebyshev_polynomials(adjacency_matrix, max_chebyshev_poly_order)

        x_input = Input(
            shape=(feature_vector_len, num_of_vertices, feature_vectors_count), name="input"
        )
        x_st_conv = SpatioTemporalConv(spatial_filters=spatial_filters, temporal_filters=temporal_filters,
                                       cheb_polynomials=cheb_polynomials,
                                       input_shape=(feature_vector_len, num_of_vertices, feature_vectors_count)
                                       )(x_input)
        x_st_conv = SpatioTemporalConv(spatial_filters=spatial_filters, temporal_filters=temporal_filters,
                                       cheb_polynomials=cheb_polynomials, )(x_st_conv)
        x_st_conv = Dropout(0.2)(x_st_conv)
        x_st_conv = Conv2DDense(kernel_size=(feature_vector_len, 1),
                                output_dim=(num_of_vertices, feature_vector_len))(x_st_conv)

        x_st_conv = Permute(dims=(2, 1))(x_st_conv)
        x_st_conv_reshaped = Reshape(target_shape=(feature_vector_len, num_of_vertices, 1))(x_st_conv)
        x_output = concatenate([x_input, x_st_conv_reshaped], axis=3)
        x_lstm = TransposeReshapeInput()(x_output)
        x_lstm = LSTM(lstm_units, return_sequences=True)(x_lstm)
        x_lstm = LSTM(lstm_units)(x_lstm)
        x_lstm = Dropout(0.2)(x_lstm)
        x_lstm = Dense(num_for_predict, activation='relu')(x_lstm)
        x_output = ReshapeOutput(output_shape=(num_of_vertices, num_for_predict))(x_lstm)

        # Instantiate an end-to-end model predicting both priority and department
        model = Model(
            inputs=[x_input],
            outputs=[x_output],
        )
        self._model = model
        self._model.summary()
