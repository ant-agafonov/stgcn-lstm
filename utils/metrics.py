import numpy as np


def mape(real_data, predicted_data):
    """
    Mean absolute percentage error.
    :param real_data: np.ndarray or int, ground truth.
    :param predicted_data: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    """
    return np.mean(np.abs(predicted_data - real_data) / (real_data + 1e-10) * 100)


def mape_in_columns(real_data, predicted_data):
    """
    Mean absolute percentage error. Averaged by rows
    :param real_data: np.ndarray, ground truth.
    :param predicted_data: np.ndarray, prediction.
    :return: int, MAPE averages on all elements of input.
    """
    return np.mean(np.abs(predicted_data - real_data) / (real_data + 1e-10) * 100, axis=0)


def rmse(real_data, predicted_data):
    """
    Mean squared error.
    :param real_data: np.ndarray or int, ground truth.
    :param predicted_data: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    """
    return np.sqrt(np.mean((predicted_data - real_data) ** 2))


def rmse_in_columns(real_data, predicted_data):
    """
    Mean squared error. Averaged by rows
    :param real_data: np.ndarray, ground truth.
    :param predicted_data: np.ndarray, prediction.
    :return: int, RMSE averages on all elements of input.
    """
    return np.sqrt(np.mean((predicted_data - real_data) ** 2, axis=0))


def mae(real_data, predicted_data):
    """
    Mean absolute error.
    :param real_data: np.ndarray or int, ground truth.
    :param predicted_data: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    """
    return np.mean(np.abs(predicted_data - real_data))


def mae_in_columns(real_data, predicted_data):
    """
    Mean absolute error. Averaged by rows
    :param real_data: np.ndarray, ground truth.
    :param predicted_data: np.ndarray, prediction.
    :return: int, MAE averages on all elements of input.
    """
    return np.mean(np.abs(predicted_data - real_data), axis=0)
