import numpy as np


def accuracy(y, y_pred):
    error_rate = ((y-y_pred) != 0).sum()/len(y)
    return 1 - error_rate
