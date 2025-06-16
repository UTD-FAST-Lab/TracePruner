from sklearn.cluster import KMeans
from metric_learn import ITML_Supervised
import numpy as np
from approach.utils import balance_training_set, balance_labeled_data_with_smote
from approach.clustering.base_clustering import BaseClusteringModel

class MPCKMeansClusterer(BaseClusteringModel):
    def __init__(self, n_clusters=2):
        super().__init__()
        self.n_clusters = n_clusters
        self.metric_learner = None
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.X_train = None

    def fit_downsample(self, X_train, train):
        self.metric_learner = ITML_Supervised(num_constraints=2000)

        # Extract only known instances
        labeled_instances = [inst for i, inst in enumerate(train) if inst.is_known()]
        X_labeled = [X_train[i] for i, inst in enumerate(train) if inst.is_known()]

        # Balance the labeled set
        balanced_labeled = balance_training_set(labeled_instances)

        # Reconstruct X and y for metric learning
        X_train_known = []
        Y_train_known = []

        for inst in balanced_labeled:
            # Find corresponding vector in X_train using instance identity
            try:
                idx = labeled_instances.index(inst)
                X_train_known.append(X_labeled[idx])
                Y_train_known.append(int(inst.get_label()))
            except ValueError:
                print("not found!")
                
        self.metric_learner.fit(X_train_known, Y_train_known) #should be fit only with labeled data 
        X_transformed = self.metric_learner.transform(X_train)

        
        self.clusterer.fit(X_transformed)


    def fit(self, X_train, train):
        self.metric_learner = ITML_Supervised(num_constraints=2000)

        X_train_known = []
        Y_train_known = []

        for i, inst in enumerate(train):
            if inst.is_known():
                X_train_known.append(X_train[i])
                Y_train_known.append(int(inst.get_label()))

        self.metric_learner.fit(X_train_known, Y_train_known) #should be fit only with labeled data TODO: balance the data before training
        X_transformed = self.metric_learner.transform(X_train)

        
        self.clusterer.fit(X_transformed)
    

    def fit_smote(self, X_train, train):
        '''smote balancing'''
        self.metric_learner = ITML_Supervised(num_constraints=2000)

        balanced_instances, X_train_known, Y_train_known = balance_labeled_data_with_smote(train, X_train)


        self.metric_learner.fit(X_train_known, Y_train_known) #should be fit only with labeled data TODO: balance the data before training
        X_transformed = self.metric_learner.transform(X_train)

        
        self.clusterer.fit(X_transformed)


    def predict(self, X_test):
        X_transformed = self.metric_learner.transform(X_test)
        cluster_ids = self.clusterer.predict(X_transformed)
        strengths = np.ones(len(X_test))
        return cluster_ids, strengths
