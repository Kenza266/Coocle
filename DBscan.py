import math
import numpy as np
from tqdm import tqdm

class DBSCAN():
    def __init__(self, eps=1, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def similarity(self, a, b):
        return len(a) + len(b) - 2 * len(set(a) & set(b))

    def euclidean_distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
        return math.sqrt(distance) 

    def _get_neighbors(self, sample_i):
        neighbors = [] 
        idxs = np.arange(len(self.X))
        for i, _sample in enumerate(self.X[idxs != sample_i]):
            distance = self.similarity(self.X[sample_i], _sample)
            #distance = self.euclidean_distance(self.X[sample_i], _sample)
            if distance < self.eps:
                neighbors.append(i)
        return np.array(neighbors)

    def _expand_cluster(self, sample_i, neighbors):
        cluster = [sample_i]
        for neighbor_i in neighbors:
            if not neighbor_i in self.visited_samples:
                self.visited_samples.append(neighbor_i)
                self.neighbors[neighbor_i] = self._get_neighbors(neighbor_i)
                if len(self.neighbors[neighbor_i]) >= self.min_samples:
                    expanded_cluster = self._expand_cluster(
                        neighbor_i, self.neighbors[neighbor_i])
                    cluster = cluster + expanded_cluster
                else:
                    cluster.append(neighbor_i)
        return cluster

    def _get_cluster_labels(self):
        labels = np.full(shape=self.X.shape[0], fill_value=len(self.clusters))
        for cluster_i, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_i
        return labels

    def predict(self, X):
        self.X = X
        self.clusters = []
        self.visited_samples = []
        self.neighbors = {}
        n_samples = np.shape(self.X)[0]
        for sample_i in tqdm(range(n_samples)):
            if sample_i in self.visited_samples:
                continue
            self.neighbors[sample_i] = self._get_neighbors(sample_i)
            if len(self.neighbors[sample_i]) >= self.min_samples:
                self.visited_samples.append(sample_i)
                new_cluster = self._expand_cluster(
                    sample_i, self.neighbors[sample_i])
                self.clusters.append(new_cluster)

        cluster_labels = self._get_cluster_labels()
        return cluster_labels