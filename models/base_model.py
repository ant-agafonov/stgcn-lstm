"""
Base class for the traffic prediction
"""
import pandas as pd
import time

from sklearn.preprocessing import MinMaxScaler
from numpy import genfromtxt, savetxt

from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop

from utils.metrics import *

__author__ = "Agafonov"


class BaseModel(object):
    def __init__(self, config):
        self._config = config
        self._scaler = None
        self._model = None
        self._data = None

        # common settings
        data_config = self._config['Data']
        hours_per_day = int(data_config['hours_per_day'])
        points_per_hour = int(self._config['Data']['points_per_hour'])
        num_for_predict = int(self._config['Data']['num_for_predict'])

        training_config = self._config['Training']
        num_of_hours = int(training_config['num_of_hours'])

        # default split proportions
        self.training_set_proportion = 0.6
        self.validation_set_proportion = 0.8

        # explicit split by lines
        self.training_split_line = None
        self.validation_split_line = None
        if config.has_option('Training', 'training_split_days') \
                and config.has_option('Training', 'validation_split_days'):
            training_split_days = int(training_config['training_split_days'])
            validation_split_days = int(training_config['validation_split_days'])
            predictions_per_day = hours_per_day * points_per_hour - num_of_hours * points_per_hour - num_for_predict + 1
            self.training_split_line = training_split_days * predictions_per_day
            self.validation_split_line = validation_split_days * predictions_per_day

    def load(self, model_name):
        self.load_model(self._model, model_name)

    def load_model(self, model, model_name):
        model.load_weights(self.get_model_path(model_name))

    def save(self, model_name):
        self.save_model(self._model, model_name)

    def save_model(self, model, model_name):
        model.save(self.get_model_path(model_name))

    def get_model_path(self, model_name):
        training_config = self._config['Training']
        params_dir = training_config['params_dir']
        model_path = params_dir + model_name + ".h5"
        return model_path

    def train(self, model_name):
        # load the dataset
        if self._data is None:
            self.read_data()
        dataset = self.create_dataset_by_config()

        train_dataset = dataset['train']['data']
        train_real_values = dataset['train']['target']

        validation_dataset = dataset['val']['data']
        validation_real_values = dataset['val']['target']

        epochs = int(self._config['Training']['epochs'])
        batch_size = int(self._config['Training']['batch_size'])
        learning_rate = float(self._config['Training']['learning_rate'])
        optimizer = Adam(lr=learning_rate)
        self._model.compile(loss='mean_squared_error', optimizer=optimizer)

        model_path = self.get_model_path(model_name)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min',
                                   save_weights_only=False)
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4,
                                           mode='min')
        history = self._model.fit(train_dataset, train_real_values, epochs=epochs, batch_size=batch_size, verbose=2,
                                  shuffle=False, validation_data=(validation_dataset, validation_real_values),
                                  callbacks=[early_stopping, mcp_save, reduce_lr_loss])
        self.load(model_name)  # load the best model

        # convert the history.history dict to a pandas DataFrame and save to csv
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = self._config['Training']['params_dir'] + "history.csv"
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        self.plot_model_architecture(self._model)

    def evaluate(self):
        if not self._model:
            raise ValueError("Model is not trained or loaded")
        # load the dataset
        if self._data is None:
            self.read_data()
        dataset = self.create_dataset_by_config()
        self.evaluate_model(self._model, dataset, 'test')

    def evaluate_model(self, model, dataset, subset_name):
        """

        :param model: model to evaluate
        :param dataset:
        :param subset_name: 'train', 'val', 'test'
        :return:
        """
        print("Evaluate the model on '{}' dataset".format(subset_name))
        subset = dataset[subset_name]['data']
        real_values = dataset[subset_name]['target']
        # make predictions and calculate errors
        start_time = time.time()
        batch_size = 1
        predicted_values = model.predict(subset, batch_size=batch_size)
        avg_time = (time.time() - start_time) * 1000 / subset.shape[0]
        print('Avg time:\t{}\trecords:\t{}'.format(avg_time, subset.shape[0]))

        predicted_values = predicted_values.reshape(-1, predicted_values.shape[2])
        test_real_values = real_values.reshape(-1, real_values.shape[2])
        self.calculate_errors(predicted_values, test_real_values)

    def calculate_errors(self, normalized_predicted_values, normalized_real_values):
        """
        Calculate and print errors
        :param normalized_predicted_values: ndarray (records, num_for_predict)
        :param normalized_real_values: ndarray (records, num_for_predict)
        :return:
        """
        predicted_values = self.denormalization(normalized_predicted_values)
        real_values = self.denormalization(normalized_real_values)

        print(self._config['Training']['model_name'])
        print('Number of predictions:\t', len(predicted_values))

        print('MAE:\t', mae(real_values, predicted_values))
        print('RMSE:\t', rmse(real_values, predicted_values))
        print('MAPE:\t', mape(real_values, predicted_values))

        print('MAE:\t', '\t'.join(map(str, mae_in_columns(real_values, predicted_values))))
        print('RMSE:\t', '\t'.join(map(str, rmse_in_columns(real_values, predicted_values))))
        print('MAPE:\t', '\t'.join(map(str, mape_in_columns(real_values, predicted_values))))

    def split_data(self, dataset, target_values):
        """
        Split the dataset into training, validation, testing sets
        :param dataset: features
        :param target_values: target
        :return: splitted dataset
        """
        if self.training_split_line is None:
            self.training_split_line = int(len(dataset) * self.training_set_proportion)
        if self.validation_split_line is None:
            self.validation_split_line = int(len(dataset) * self.validation_set_proportion)

        training_set = dataset[:self.training_split_line]
        self._scaler = MinMaxScaler(feature_range=(0, 1)).fit(training_set.reshape(-1, 1))

        training_set = self.normalization(training_set)
        training_set_target = self.normalization(target_values[:self.training_split_line])

        validation_set = self.normalization(dataset[self.training_split_line:self.validation_split_line])
        validation_set_target = self.normalization(target_values[self.training_split_line:self.validation_split_line])

        testing_set = self.normalization(dataset[self.validation_split_line:])
        testing_set_target = self.normalization(target_values[self.validation_split_line:])

        all_data = {
            'train': {
                'data': training_set,
                'target': training_set_target,
            },
            'val': {
                'data': validation_set,
                'target': validation_set_target,
            },
            'test': {
                'data': testing_set,
                'target': testing_set_target
            }
        }
        print('training data: {}'.format(training_set.shape))
        print('validation data: {}'.format(validation_set.shape))
        print('testing data: {}'.format(testing_set.shape))
        print('all samples len: {}'.format(len(dataset)))
        return all_data

    def read_data(self):
        """
        Read the raw data as the matrix with the shape (records count, links/sensors count)
        :return:
        """
        data_config = self._config['Data']
        data_filename = data_config['data_filename']
        data_type = data_config['data_type']
        data_seq = None
        if data_type == 'npz':
            # Zipped PeMS data with the shape (records count, links/sensors count, features count)
            used_features_index = int(data_config['used_features_index'])
            data_seq = np.load(data_filename)['data'][:, :, used_features_index]
        elif data_type == 'csv_matrix':
            # PeMS data: csv matrix data with the shape (records count, links/sensors count)
            data_seq = genfromtxt(data_filename, delimiter=',')
        self._data = data_seq

    def normalization(self, data_set):
        scaled_data = self._scaler.transform(data_set.reshape(-1, 1))
        return scaled_data.reshape(data_set.shape)

    def denormalization(self, target_norm):
        unscaled_data = self._scaler.inverse_transform(target_norm.reshape(-1, 1))
        return unscaled_data.reshape(target_norm.shape)

    def create_dataset_by_config(self):
        """
        dataset has the following shape (records, links, features)
        :return:
        """
        # common settings
        data_config = self._config['Data']
        days_per_week = int(data_config['days_per_week'])
        hours_per_day = int(data_config['hours_per_day'])
        records_per_hour = int(data_config['points_per_hour'])
        num_for_predict = int(data_config['num_for_predict'])

        # training settings
        training_config = self._config['Training']
        num_of_weeks = int(training_config['num_of_weeks'])
        num_of_days = int(training_config['num_of_days'])
        num_of_hours = int(training_config['num_of_hours'])

        # load the dataset
        records_per_day = records_per_hour * hours_per_day
        days_count = self._data.shape[0] // records_per_day
        # reshape to (days, records per day, links)
        data_by_days = self._data.reshape(days_count, records_per_day, self._data.shape[1])
        dataset = []
        target_values = []
        for day_index in range(days_count):
            for current_index in range(records_per_day):
                features = self.get_features(data_by_days, num_of_weeks, num_of_days, num_of_hours,
                                             day_index, current_index, num_for_predict,
                                             days_per_week, hours_per_day, records_per_hour)
                if not features:
                    continue  # not enough data (for example, cannot use statistics for previous weeks)

                # concatenate all features into one feature vector
                feature_vector = np.dstack(features[0:len(features) - 1])
                target = features[len(features) - 1]
                target = target.transpose((1, 0))

                dataset.append(feature_vector)
                target_values.append(target)

        dataset = np.stack(dataset, axis=0)
        target_values = np.stack(target_values, axis=0)
        return self.split_data(dataset, target_values)

    def get_features(self, data, num_of_weeks, num_of_days, num_of_hours,
                     day_index, time_index, num_for_predict,
                     days_per_week=7, hours_per_day=24, records_per_hour=6):
        """
        Convert data into feature vectors (feature vector len, feature vectors count, links count)
        Let the current time be t - 1 (label_start_idx is t). Consider features of three types:
        1) Current sequence:
        {t - num_of_hours * records_per_hour, t - num_of_hours * records_per_hour + 1, ..., t - 1 }
        2) Day / week sequence:
        {t - records_per_day(week) - num_of_hours * records_per_hour, ..., t - records_per_day(week) - 1}
        Number of such feature vectors is num_of_days (num_of_weeks)
        3) Day / week shifted sequence (shift is equal to num_for_predict):
        {t - records_per_day(week) - num_of_hours * records_per_hour + num_for_predict,
            ..., t - records_per_day(week) + num_for_predict - 1}
        This feature vector allows to use statistics in times (t, ..., t + num_for_predict - 1) a day ago, for example.

        :param data: np.ndarray, shape is (days, time_sequence_length, num_of_vertices)
        :param num_of_weeks: int
        :param num_of_days: int
        :param num_of_hours: int
        :param day_index: current day
        :param time_index: int, the first time index of predicting target
        :param num_for_predict: int, the number of records will be predicted for each sample
        :param days_per_week: int, some datasets have only weekday records
        :param hours_per_day:
        :param records_per_hour: int, default 6, number of records per hour
        :return:
            features as described above + target: [np.ndarray], shape is (num_for_predict, num_of_vertices)
        """
        times_length = data.shape[1]
        feature_size = num_of_hours * records_per_hour
        if time_index - feature_size < 0:  # not enough records to create a feature vector
            return None
        if time_index + num_for_predict - 1 >= times_length:  # not enough records to predict
            return None

        features = []
        # weeks features, count = num_of_weeks * 2 (types = 2)
        for i in range(1, num_of_weeks + 1):
            start_day_index = day_index - i * days_per_week
            # type 2
            feature = data[start_day_index, time_index - feature_size:time_index, :]
            features.append(feature)
            # type 3
            feature = data[start_day_index, time_index - feature_size + num_for_predict:time_index + num_for_predict, :]
            features.append(feature)

        # days features, count = num_of_days * 2 (types = 2)
        for i in range(1, num_of_days + 1):
            start_day_index = day_index - i
            # type 2
            feature = data[start_day_index, time_index - feature_size:time_index, :]
            features.append(feature)
            # type 3
            feature = data[start_day_index, time_index - feature_size + num_for_predict:time_index + num_for_predict, :]
            features.append(feature)

        # hour features, count = 1
        feature = data[day_index, time_index - feature_size: time_index, :]
        features.append(feature)

        # target
        target = data[day_index, time_index: time_index + num_for_predict, :]
        features.append(target)
        return features

    def plot_model_architecture(self, model):
        # plot the model architecture
        try:
            model_architecture = self._config['Training']['params_dir'] + 'model.pdf'
            plot_model(model, to_file=model_architecture, show_shapes=True)
        except:
            print("Model is not visualized")

    def save_data(self, data, file_name):
        file_path = self._config['Training']['params_dir'] + file_name
        savetxt(file_path, data)
