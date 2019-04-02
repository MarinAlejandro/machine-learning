import numpy as np


def gini(targets):
    classes = np.unique(targets)
    probabilities = [len(targets[targets == class_]) / len(targets) for class_ in classes]
    gini = sum([p * (1 - p) for p in probabilities])
    return gini