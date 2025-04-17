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

class CDistFallback2(FallbackStrategy):
    
    def __init__(self, epsilon=1e-8):
        self.true_points = None
        self.epsilon = epsilon  # for numerical stability

    def fit(self, X_train, y_train):
        # Store only the feature vectors of known True-labeled instances
        self.true_points = X_train[y_train == 1]

    def predict(self, X_test):
        if self.true_points is None or len(self.true_points) == 0:
            raise ValueError("No true-labeled points available for fallback.")

        distances = cdist(X_test, self.true_points)  # shape: (n_test, n_true)
        labels = []

        for row in distances:
            if np.any(row <= self.epsilon):  # check if any true point is identical
                labels.append(1)  # assign True
            else:
                labels.append(0)  # assign False

        return labels

    def predict_proba(self, X_test):
        # Optional: confidence = 1.0 if match found, 0.0 otherwise
        distances = cdist(X_test, self.true_points)
        return np.array([1.0 if np.any(row <= self.epsilon) else 0.0 for row in distances])