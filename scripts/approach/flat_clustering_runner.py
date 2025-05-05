import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from approach.utils import evaluate_fold, write_metrics_to_csv, plot_clusters_by_label
from collections import defaultdict
from scipy.spatial.distance import cdist


class FlatClusteringRunner:
    def __init__(self, instances, clusterer, output_dir, use_trace=False, use_semantic=False, use_static=True):
        self.instances = instances
        self.clusterer = clusterer
        self.output_dir = output_dir
        self.use_trace = use_trace
        self.use_semantic = use_semantic
        self.use_static = use_static

        self.labeled = [i for i in instances if i.is_known()]
        self.unknown = [i for i in instances if not i.is_known()]

    def get_features(self, insts):
        feature_vecs = []
        for inst in insts:
            feature = []
            if self.use_semantic:
                code = inst.get_semantic_features()
                if code is None or len(code) != 768:
                    print(f"Semantic features not found for {inst.get_id()}")
                    code = [0.0] * 768
                feature_vecs.append(code)
                # feature += code
            if self.use_static:
                struct = inst.get_static_featuers()
                if struct is None or len(struct) != 11:
                    struct = [0.0] * 11
                feature_vecs.append(struct)
                # feature += struct
            if self.use_trace:
                trace = inst.get_trace_features()
                if trace is None or len(trace) != 128:
                    trace = [0.0] * 128
                # feature += trace
                feature_vecs.append(trace)
            # feature_vecs.append(feature)
        return np.array(feature_vecs)

    def run(self):
        instances = self.instances
        X = self.get_features(instances)
        print("found %d instances" % len(X))

        if self.use_static:
            X_scaled = StandardScaler().fit_transform(X)
        elif self.use_semantic or self.use_trace:
            from sklearn.decomposition import FastICA
            X_scaled = FastICA(n_components=50).fit_transform(X)
        else:
            X_scaled = X
        # from sklearn.decomposition import FastICA
        # X_scaled = FastICA(n_components=50).fit_transform(X)

        true_instances = [i for i in self.labeled if i.get_label() is True]
        true_labeled, _ = train_test_split(true_instances, test_size=0.5, random_state=42)
        seed_indices = [i for i, inst in enumerate(instances) if inst in true_labeled]
        y_seed = np.zeros(len(instances))
        for i in seed_indices:
            y_seed[i] = 1

        self.clusterer.fit(X_scaled)

        # plot the clusters
        cluster_ids = self.clusterer.clusterer.labels_
        probs = self.clusterer.clusterer.probabilities_
        # plot_clusters_by_label(self.instances, X_scaled, cluster_ids, probs, f"{self.output_dir}/cluster_plot.png")

        cluster_labels = self.label_clusters(y_seed, cluster_ids)



        seed_X = X_scaled[seed_indices]

        predictions = []
        for i, cid in enumerate(cluster_ids):
            if cid != -1:
                pred = cluster_labels.get(cid, 0)  # default to 0 if cluster not in label map
            else:
                pred = self.fallback(seed_X, X_scaled[i])
            predictions.append(pred)
            instances[i].set_predicted_label(pred)
            instances[i].set_confidence(1.0)

        y_true = [1 if inst.get_label() else 0 for inst in self.labeled]
        y_pred = [1 if inst.get_predicted_label() else 0 for inst in self.labeled]
        overall = evaluate_fold(y_true, y_pred)

        gt_instances = [i for i in self.instances if i.ground_truth is not None and not i.is_known()]
        gt_y_true = [int(i.ground_truth) for i in gt_instances]
        gt_y_pred = [int(i.get_predicted_label()) for i in gt_instances]
        gt_metrics = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}
        for k, v in gt_metrics.items():
            overall[f"gt_{k}"] = v

        print("\n=== Flat Clustering Evaluation ===")
        print(f"Precision: {overall['precision']:.3f}, Recall: {overall['recall']:.3f}, F1: {overall['f1']:.3f}")
        print(f"TP: {overall['TP']} | FP: {overall['FP']} | TN: {overall['TN']} | FN: {overall['FN']}")

        # write_metrics_to_csv([overall], f"{self.output_dir}/trace_n2v_sum_tfidf2.csv")

    def fallback(self, seed_X, point):
        distances = cdist([point], seed_X)[0]
        return int(np.any(distances == 0.0))

    def label_clusters(self, y_seed, cluster_ids):
        cluster_to_labels = defaultdict(list)
        for idx, (cid, label) in enumerate(zip(cluster_ids, y_seed)):
            if cid != -1 and label == 1:
                cluster_to_labels[cid].append(1)
        return {cid: 1 for cid in cluster_to_labels}
