import numpy as np
from trees import StumpBinary
from metrics import accuracy


class AdaBoostBinary:
    
    def __init__(self, n_classifiers, sign_y=False):
        self.n_classifiers = n_classifiers
        self.sign_y = sign_y
        self.beta = []
        self.classifiers = []
        
    def train(self, x, y):
        n_samples = x.shape[0]
        weights = np.full(n_samples, 1 / n_samples)
            
        for i in range(self.n_classifiers):
            clf_i = StumpBinary(weights=weights)
            clf_i.train(x, y)
            y_pred = clf_i.predict(x)
            accuracy_i = accuracy(y, y_pred)
            # remove hard coded 10 to symbolize same data
            beta_i = 0.5 * np.log(accuracy_i / (1 - accuracy_i)) if accuracy_i < 1 else 10
            coef = np.ones(n_samples)
            coef[y != y_pred] = -1
            coef *= beta_i
            weights *= np.exp(-coef)
            weights /= weights.sum()
            self.beta.append(beta_i)
            self.classifiers.append(clf_i)
            
    def predict(self, x):
        y_total = 0
        for clf, beta in zip(self.classifiers, self.beta):
            beta = beta if beta != 10 else 1
            y_total += beta * clf.predict(x)
        if self.sign_y:
            y_total = np.sign(y_total).astype(int)
        return y_total
    

class AdaBoost:
    
    def __init__(self, n_classifiers, sign_y=False):
        self.n_classifiers = n_classifiers
        self.sign_y = sign_y
        self.classifiers = None
        self.beta = None
        
    def train(self, x, y):
        classes = np.unique(y)
        self.classifiers = []
        for class_ in classes:
            y_binary = y.copy()
            y_binary = (y_binary == class_) * 2 - 1
            clf_ = AdaBoostBinary(self.n_classifiers, self.sign_y)
            clf_.train(x, y_binary)
            self.classifiers.append(clf_)
            
    def predict(self, x):
        y_pred = []
        for clf_ in self.classifiers:
            y_pred_i = clf_.predict(x)
            y_pred.append(y_pred_i)
        y_pred = np.array(y_pred)
        y_total = np.argmax(y_pred, axis=0)
        return y_total
