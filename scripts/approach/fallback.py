from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
import numpy as np

class FallbackStrategy:
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

class KNNFallback(FallbackStrategy):
    def __init__(self, k=5):
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X, y):
        self.knn.fit(X, y)

    def predict(self, X):
        return self.knn.predict(X)

    def predict_proba(self, X):
        return self.knn.predict_proba(X).max(axis=1)

class CDistFallback(FallbackStrategy):
    def fit(self, X, y, cluster_ids, cluster_labels):
        self.X = X
        self.y = y
        self.cluster_ids = cluster_ids
        self.cluster_labels = cluster_labels

    def predict(self, X_test):
        distances = cdist(X_test, self.X)
        nearest_idx = np.argmin(distances, axis=1)
        
        labels = []

        for indx in nearest_idx:
            cid = self.cluster_ids[indx]
            if cid == -1 or cid not in self.cluster_labels:
                labels.append(0)
            else:
                labels.append(self.cluster_labels[cid])

        return labels

    def predict_proba(self, X_test):
        return np.ones(len(X_test)) * 0.5  # placeholder
