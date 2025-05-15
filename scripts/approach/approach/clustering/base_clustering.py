from abc import ABC, abstractmethod

class BaseClusteringModel(ABC):

    def __init__(self):
        self.cluster_labels = {}
    
    @abstractmethod
    def fit(self, X_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        """Returns cluster_id, confidence"""
        pass

    def get_cluster_label(self, cluster_id):
        return self.cluster_labels.get(cluster_id, None)

    def label_clusters(self, labels, cluster_ids, labeler):
        self.cluster_labels = labeler.label_clusters(labels, cluster_ids)
