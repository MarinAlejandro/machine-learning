import numpy as np
from utils import gini
import math


class StumpBinary:

    def __init__(self, weights=None):
        self.weights = weights
        self.max_class_left = None
        self.max_class_right = None
        self.threshold = None
        self.feature = None 

    def train(self, x, y):
        # Threshold can be based in the middle point or in data.
        # Threshold is chosen among data points except last one.
        # <= for left child
        # Values have to be sorted (nlogn) but to calculate information gain we can take advantage of the sort.
        # Not implemented last comment yet.

        n_classes = len(np.unique(y))
        max_info = 0
        max_info_feature = 0
        max_info_threshold = 0
        n_samples = x.shape[0]
        n_features = x.shape[1]
        
        if self.weights is None:
            self.weights = np.full(n_samples, 1.0 / n_samples)
        
        assert (n_classes == 2)
        assert(math.isclose(self.weights.sum(), 1))
        assert(len(self.weights) == n_samples)

        for feature in range(n_features):
            thresholds_bag = np.sort(np.unique(x[:, feature]))[:-1]
            for threshold in thresholds_bag:
                left_mask = x[:, feature] <= threshold
                left_child = y[left_mask]
                right_child = y[~left_mask]
                frac_left = self.weights[left_mask].sum()
                info_gain = gini(y) - (frac_left * gini(left_child) + (1 - frac_left) * gini(right_child))
                if info_gain > max_info:
                    max_info = info_gain
                    max_info_threshold = threshold
                    max_info_feature = feature

        self.threshold = max_info_threshold
        self.feature = max_info_feature

        left_mask = x[:, self.feature] <= self.threshold
        (left_values, left_counts) = np.unique(y[left_mask], return_counts=True)
        (right_values, right_counts) = np.unique(y[~left_mask], return_counts=True)

        self.max_class_left = left_values[np.argmax(left_counts)]
        self.max_class_right = right_values[np.argmax(right_counts)]

    def predict(self, x):
        n_samples = x.shape[0]
        y_pred = np.zeros(n_samples)
        left_mask = x[:, self.feature] <= self.threshold
        y_pred[left_mask] = self.max_class_left
        y_pred[~left_mask] = self.max_class_right
        return y_pred


class Stump:

    def __init__(self, weights=None):
        self.classifiers = []
        self.weights = weights

    def train(self, x, y):
        classes = np.unique(y)
        for class_ in classes:
            # y_binary = (preds == class_) * 2 - 1
            y_binary = y.copy()
            class_ *= 2  # case classes are -1 and 1
            y_binary *= 2
            y_binary[y_binary != class_] = -1
            y_binary[y_binary == class_] = 1
            clf_i = StumpBinary(self.weights)
            clf_i.train(x, y_binary)
            self.classifiers.append(clf_i)

    def predict(self, x):
        n_samples = x.shape[0]
        y_pred = []
        for clf_i in self.classifiers:
            y_pred_i = clf_i.predict(x)
            y_pred.append(y_pred_i)
        y_total = np.full(n_samples, np.NINF)
        y_total_i = np.zeros(n_samples)
        # assumed class 0, 1, 2 so index is the class
        for i in range(len(self.classifiers)):
            y_total_i[y_pred[i] - y_total > 0] = i
            y_total = np.maximum(y_total, y_pred[i])
        return y_total_i
