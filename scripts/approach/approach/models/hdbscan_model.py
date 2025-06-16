import hdbscan
from approach.clustering.base_clustering import BaseClusteringModel

class HDBSCANClusterer(BaseClusteringModel):
    def __init__(self, min_cluster_size=5, min_samples=5, cluster_selection_method='eom', metric='euclidean', alpha=1.0):
        super().__init__()
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric=metric,
            alpha=alpha,
            prediction_data=True)

    def fit(self, X_train):
        self.clusterer.fit(X_train)

    def predict(self, X_test):
        cluster_ids, strengths = hdbscan.approximate_predict(self.clusterer, X_test)
        return cluster_ids, strengths
