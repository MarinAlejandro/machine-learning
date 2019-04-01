import numpy as np
from scipy.stats import multivariate_normal
from kmeans import KMeans


class GMM:
    def __init__(self, n_clusters, n_iterations=100, regu=1e-6, init='kmeans'):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.means = []
        self.covariances = []
        self.weights = []
        self.regu = regu
        self.init = init
        self.z = None
        self.n_dimensions = None
        self.n_samples = None

    def _init_parameters(self, x):
        mean_variance = np.zeros((self.n_dimensions, self.n_dimensions))
        for n in range(self.n_samples):
            mean_variance += np.dot((x[n] - x.mean(axis=0)).reshape(-1, 1), (x[n] - x.mean(axis=0)).reshape(-1, 1).T)
        mean_variance /= self.n_samples
        for k in range(self.n_clusters):
            self.weights.append(1 / self.n_clusters)
            self.covariances.append(mean_variance)
        if self.init == 'kmeans':
            k_means = KMeans(n_clusters=self.n_clusters)
            k_means.train(x)
            self.means = k_means.get_means()
        else:
            idx_means = np.random.choice(self.n_samples, self.n_clusters, replace=False)
            self.means = [x[idx_means[k]] for k in range(self.n_clusters)]

    def _expectation(self, x):
        z = self.z
        for k in range(self.n_clusters):
            for n in range(self.n_samples):
                params = {
                    'mean': self.means[k],
                    'cov': self.covariances[k] + np.identity(self.n_dimensions) * self.regu,
                    'x': x[n]
                }
                g = self.weights[k] * multivariate_normal.pdf(**params)
                z[n, k] = g + np.finfo('float64').eps
        z /= z.sum(axis=1, keepdims=True)
        return z

    def _maximization(self, x):
        z = self.z
        n_k = np.sum(z, axis=0)
        means = []
        covariances = []
        weights = []

        for k in range(self.n_clusters):
            mean = 0
            for n in range(self.n_samples):
                mean += z[n, k] * x[n]
            mean /= n_k[k]
            means.append(mean)

        for k in range(self.n_clusters):
            cov = np.zeros((self.n_dimensions, self.n_dimensions))
            for n in range(self.n_samples):
                cov += z[n, k] * np.dot((x[n] - means[k]).reshape(-1, 1), (x[n] - means[k]).reshape(-1, 1).T)
            cov /= n_k[k]
            covariances.append(cov)

        for k in range(self.n_clusters):
            weights.append(n_k[k] / self.n_samples)
        return means, covariances, weights

    def train(self, x):
        self.n_samples = x.shape[0]
        self.n_dimensions = x.shape[1]
        self.z = np.zeros((self.n_samples, self.n_clusters))
        self._init_parameters(x)
        for i in range(self.n_iterations):
            self.z = self._expectation(x)
            self.means, self.covariances, self.weights = self._maximization(x)

    def predict(self, x):
        n_samples = x.shape[0]
        y_test = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            for n in range(n_samples):
                params = {
                    'mean': self.means[k],
                    'cov': self.covariances[k] + np.identity(self.n_dimensions) * self.regu,
                    'x': x[n]
                }
                g = self.weights[k] * multivariate_normal.pdf(**params)
                y_test[n, k] = g
        return np.argmax(y_test, axis=1)