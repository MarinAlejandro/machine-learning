from kernels import gauss_kernel, polynomial_kernel
import numpy as np
from scipy.optimize import minimize


class SVC:
    def __init__(self, C=1, kernel=polynomial_kernel, gamma=1, degree=2):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.n_classes = None
        self.classifiers = []

    def train(self, x, y):
        self.n_classes = np.unique(y)
        for class_i in self.n_classes:
            class_i *= 2
            y_binary = y.copy()
            y_binary *= 2
            y_binary[y_binary != class_i] = -1
            y_binary[y_binary == class_i] = 1
            clf_i = SVCBinary(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree)
            clf_i.train(x, y_binary)
            self.classifiers.append(clf_i)

    def predict(self, x, per_class=False):
        y_pred = []
        for clf_i in self.classifiers:
            y_pred_i = clf_i.predict(x)
            # y_pred_i /= y_pred_i.max() // can be tried to compare different classes and improve predictions
            y_pred.append(y_pred_i)
        y_total = np.zeros(y_pred[0].shape)
        y_total_i = np.zeros(y_pred[0].shape)
        for i in range(len(self.n_classes)):
            y_total_i[y_pred[i] - y_total > 0] = i
            y_total = np.maximum(y_total, y_pred[i])
        return y_pred if per_class else y_total_i


class SVCBinary:
    def _kernel_matrix(self, x, kernel_fun):
        n_samples = x.shape[0]
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = kernel_fun(x[i], x[j], self.gamma, self.degree)
        return kernel_matrix

    @staticmethod
    def _q_matrix(kernel_matrix, y):
        n_samples = y.shape[0]
        q_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                q_matrix[i, j] = y[i] * y[j] * kernel_matrix[i, j]
        return q_matrix

    def _loss(self, alpha):
        loss = -alpha.sum() + 0.5 * (alpha.T @ self.q_matrix @ alpha)
        return loss

    def __init__(self, C=1, kernel=polynomial_kernel, gamma=1, degree=2):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.kernel_matrix = None
        self.q_matrix = None
        self.alpha = None
        self.alpha_y = None
        self.w = None
        self.b = None
        self.n_samples = None

    def train(self, x, y):
        self.n_samples = x.shape[0]
        self.alpha = np.zeros(self.n_samples)
        self.kernel_matrix = self._kernel_matrix(x, self.kernel)
        self.q_matrix = self._q_matrix(self.kernel_matrix, y)

        bounds_alpha = tuple((0, self.C) for n in range(self.n_samples))

        def constraint_sum(alpha):
            return (alpha * y).sum()

        constraint = ({'type': 'eq', 'fun': constraint_sum})
        self.alpha = minimize(self._loss, self.alpha, bounds=bounds_alpha, constraints=constraint)['x']

        threshold = min(self.C * 1e-3, 1e-3)
        self.alpha[self.alpha < threshold] = 0
        self.alpha_y = self.alpha * y
        self.b = y[0] - self.alpha_y.T @ self.kernel_matrix[:, 0]

    def predict(self, x, round_y=False):
        n_samples = x.shape[0]
        y_predicted = np.zeros(n_samples)
        for n in range(n_samples):
            y_predicted[n] = self.alpha_y.T @ self.kernel_matrix[:, n] + self.b
        if round_y:
            y_predicted = (y_predicted > 0).astype(int) * 2 - 1
        return y_predicted
