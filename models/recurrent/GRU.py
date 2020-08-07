"""
The recurrent model with 2 GRU layers
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.layers import GRU
from models.recurrent.layers import *

from models.base_model import BaseModel

__author__ = "Agafonov"


class GRUModel(BaseModel):
    def __init__(self, config):
        """
        Initialize the model
        """
        BaseModel.__init__(self, config)

        # feature vector settings
        data_config = self._config['Data']
        points_per_hour = int(data_config['points_per_hour'])
        num_for_predict = int(data_config['num_for_predict'])
        num_of_vertices = int(data_config['num_of_vertices'])

        training_config = self._config['Training']
        num_of_weeks = int(training_config['num_of_weeks'])
        num_of_days = int(training_config['num_of_days'])
        num_of_hours = int(training_config['num_of_hours'])

        feature_vector_len = num_of_hours * points_per_hour
        feature_vectors_count = num_of_weeks * 2 + num_of_days * 2 + 1
        lstm_units = 64

        model = Sequential()
        model.add(InputLayer(input_shape=(feature_vector_len, num_of_vertices, feature_vectors_count)))
        model.add(TransposeReshapeInput())
        model.add(GRU(lstm_units, return_sequences=True))
        model.add(GRU(lstm_units))
        model.add(Dropout(0.2))
        model.add(Dense(num_for_predict, activation='relu'))
        model.add(ReshapeOutput(output_shape=(num_of_vertices, num_for_predict)))
        self._model = model
        self._model.summary()
