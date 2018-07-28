import numpy as np


def abolute_error_vector(y_true, y_pred):
    return np.abs(y_pred - y_true)


def squared_error_vector(y_true, y_pred):
    return (y_true - y_pred) ** 2
