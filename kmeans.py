import numpy as np
from numpy import linalg


class KMeans:

    def __init__(self, n_clusters, max_iterations=1000):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.means = []
        self.clusters = []
        self.converged = False

    def train(self, x):
        n_samples = x.shape[0]
        idx_means = np.random.choice(x.shape[0], self.n_clusters, replace=False)

        for k in range(self.n_clusters):
            self.means.append(x[idx_means[k]])

        for i in range(self.max_iterations):
            new_clusters = []
            for n in range(n_samples):
                new_clusters.append(np.argmin([linalg.norm(x[n] - self.means[k]) for k in range(self.n_clusters) if
                                               not np.isnan(self.means[k]).any()]))
            new_clusters = np.array(new_clusters)
            if np.array_equal(self.clusters, new_clusters):
                self.converged = True
                # print(f'Converged at step: {i}')
                break
            self.clusters = new_clusters

            new_means = []
            for k in range(self.n_clusters):
                x_k = x[self.clusters == k]
                mean = self.means[k] if x_k.size == 0 else x_k.mean(axis=0)
                new_means.append(mean)
            self.means = new_means

    def predict(self, x):
        n_samples = x.shape[0]
        new_clusters = []
        for n in range(n_samples):
            new_clusters.append(np.argmin([linalg.norm(x[n] - self.means[k]) for k in range(self.n_clusters) if
                                           not np.isnan(self.means[k]).any()]))
        new_clusters = np.array(new_clusters)
        return new_clusters

    def get_means(self):
        return self.means
