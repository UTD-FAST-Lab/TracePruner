import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from approach.utils import evaluate_fold, write_results_to_csv, write_metrics_to_csv, split_folds
from collections import defaultdict

class ClusteringRunner:
    def __init__(self, instances, clusterer, fallback, labeler, output_dir, use_trace=False, use_semantic=False, use_static=True, use_fallback=True, train_with_unknown=False):
        self.instances = instances
        self.clusterer = clusterer
        self.fallback = fallback
        self.labeler = labeler
        self.output_dir = output_dir
        self.use_trace = use_trace
        self.use_semantic = use_semantic
        self.use_static = use_static
        self.use_fallback = use_fallback
        self.train_with_unknown = train_with_unknown
        self.labeled = [i for i in self.instances if i.is_known()]
        self.unknown = [i for i in self.instances if not i.is_known()]

    def get_features(self, insts):
        
        feature_vecs = []

        for inst in insts:
            if self.use_semantic:
                code = inst.get_semantic_features()
                if code is None or len(code) != 768:
                    code = [0.0] * 768
                feature_vecs.append(code)
            elif self.use_static:
                struct = inst.get_static_featuers()
                if struct is None or len(struct) != 11:
                    struct = [0.0] * 11
                feature_vecs.append(struct)

            
        return np.array(feature_vecs)
            

    def run(self):


        folds = split_folds(self.labeled, self.unknown, self.train_with_unknown)

        all_metrics = []
        all_eval = []
        unk_labeled_true = 0
        unk_labeled_false = 0

        for fold, (train, test) in enumerate(folds, 1):
            print(f"\n=== Fold {fold} ===")

            X_train = self.get_features(train)
            y_train = np.array([1 if i.get_label() else 0 for i in train])
            X_test = self.get_features(test)

            if self.use_static:
                scaler = StandardScaler()
                X_all = scaler.fit_transform(np.vstack((X_train, X_test)))
                X_train_scaled = X_all[:len(X_train)]
                X_test_scaled = X_all[len(X_train):]
            
            elif self.use_semantic:
                # reduction of dimensions with pca
                from sklearn.decomposition import PCA
                pca = PCA(n_components=50)
                X_all = pca.fit_transform(np.vstack((X_train, X_test)))
                X_train_scaled = X_all[:len(X_train)]
                X_test_scaled = X_all[len(X_train):]



            self.clusterer.fit(X_train_scaled) #hdbscan 
            self.clusterer.label_clusters(y_train, self.clusterer.clusterer.labels_, self.labeler) #hdbscan {any_normal:y_trian, majority: train}

            # self.clusterer.fit_smote(X_train_scaled, train) #mpckmeans
            # self.clusterer.label_clusters(train, self.clusterer.clusterer.labels_, self.labeler) #mpckmeans

            # if fold == 1:
            #     self.print_cluster_distribution(train, self.clusterer.clusterer.labels_)

            cluster_ids, strengths = self.clusterer.predict(X_test_scaled)

            if self.use_fallback:
                # self.fallback.fit(X_train_scaled, y_train, self.clusterer.clusterer.labels_, self.clusterer.cluster_labels) #dist1
                self.fallback.fit(X_train_scaled, y_train) #dist2
                fallback_preds = self.fallback.predict(X_test_scaled)
                fallback_conf = self.fallback.predict_proba(X_test_scaled)

            predictions, confidences = [], []
            for i, cid in enumerate(cluster_ids):
                if cid != -1 and cid in self.clusterer.cluster_labels:
                    predictions.append(self.clusterer.cluster_labels[cid])
                    confidences.append(strengths[i])
                elif self.use_fallback:
                    predictions.append(fallback_preds[i])
                    confidences.append(fallback_conf[i])
                else:
                    predictions.append(0)
                    confidences.append(0.0)

            for inst, pred, conf in zip(test, predictions, confidences):
                inst.set_predicted_label(pred)
                inst.set_confidence(conf)

            # y_true = [1 if inst.get_label() else 0 for inst in test]
            # y_pred = predictions[:len(test)]
            # metrics = evaluate_fold(y_true, y_pred)
            res = self.evaluate(test)
            metrics = res[0]
            metrics["unk_labeled_true"] = res[1]
            metrics["unk_labeled_false"] = res[2]
            metrics["unk_labeled_all"] = res[1] + res[2]

            gt_metrics = res[3]
            if gt_metrics:
                for k, v in gt_metrics.items():
                    metrics[f"gt_{k}"] = v
            # metrics["gt_count"] = len(gt_metrics.get("TP", []))  # fallback to 0 if empty

            metrics["fold"] = fold

            unk_labeled_true += res[1]
            unk_labeled_false += res[2]
            all_metrics.append(metrics)
            all_eval.extend(test)


        # Overall
        y_all_true = [1 if inst.get_label() else 0 for inst in self.labeled]
        y_all_pred = [1 if inst.get_predicted_label() else 0 for inst in self.labeled]
        overall = evaluate_fold(y_all_true, y_all_pred)
        overall["unk_labeled_true"] = unk_labeled_true
        overall["unk_labeled_false"] = unk_labeled_false
        overall["unk_labeled_all"] = unk_labeled_false + unk_labeled_true
        
        # Add evaluation on manually labeled unknowns
        gt_instances = [i for i in self.instances if i.ground_truth is not None and not i.is_known()]
        gt_y_true = [int(i.ground_truth) for i in gt_instances]
        gt_y_pred = [int(i.get_predicted_label()) for i in gt_instances]
        gt_metrics = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}

        for k, v in gt_metrics.items():
            overall[f"gt_{k}"] = v
        # overall["gt_count"] = len(gt_y_true)
        
        overall["fold"] = "overall"
        all_metrics.append(overall)

        print("\n=== RF Overall ===")
        print(f"Precision: {overall['precision']:.3f}, Recall: {overall['recall']:.3f}, F1: {overall['f1']:.3f}")
        print(f"TP: {overall['TP']} | FP: {overall['FP']} | TN: {overall['TN']} | FN: {overall['FN']}")


        # write_results_to_csv(all_eval, f"{self.output_dir}/cluster_results.csv")
        write_metrics_to_csv(all_metrics, f"{self.output_dir}/fold_metrics_dist_plelog.csv")



    def evaluate(self, test):
        y_true = []
        y_pred = []
        
        true = 0
        false = 0

        # For GT-labeled unknowns
        gt_y_true = []
        gt_y_pred = []

        for inst in test:
            pred = int(inst.get_predicted_label())

            if inst.is_known():
                y_true.append(int(inst.get_label()))
                y_pred.append(pred)
            else:
                if pred == 1:
                    true += 1
                elif pred == 0:
                    false += 1

                if inst.ground_truth is not None:
                    gt_y_true.append(int(inst.ground_truth))
                    gt_y_pred.append(pred)
            
        
        print("all: ", true+false)
        print("true: ", true)
        print("false: ", false)
        print("Manually labeled unknowns:", len(gt_y_true))

        eval_main = evaluate_fold(y_true, y_pred)
        eval_gt = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}

        print(eval_gt)
        
        return eval_main, true, false, eval_gt
    

    def print_cluster_distribution(self, train, cluster_ids):
        '''print in each cluster how many true, false and unknowns are there'''

        cluster_2_data = defaultdict(list)

        for inst, cid in zip(train, cluster_ids):
            cluster_2_data[cid].append(inst)
        

        with open(f'{self.output_dir}/cluster_distibution_balanced.txt', 'w') as f:
            for cid, val in cluster_2_data.items():
                true = 0
                false = 0
                unk = 0
                for inst in val:
                    if not inst.is_known():
                        unk += 1
                    else:
                        if inst.get_label() is True:
                            true += 1
                        else:
                            false += 1
                
                f.write(f"cid: {cid} -> true: {true}\t false: {false}\t unk: {unk}\n ")

            
            
