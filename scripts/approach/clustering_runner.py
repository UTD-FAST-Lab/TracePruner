import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from approach.utils import evaluate_fold, write_results_to_csv, write_metrics_to_csv, split_folds

class ClusteringRunner:
    def __init__(self, instances, clusterer, fallback, labeler, output_dir, use_trace=False, use_fallback=True, train_with_unknown=False):
        self.instances = instances
        self.clusterer = clusterer
        self.fallback = fallback
        self.labeler = labeler
        self.output_dir = output_dir
        self.use_trace = use_trace
        self.use_fallback = use_fallback
        self.train_with_unknown = train_with_unknown
        self.labeled = [i for i in self.instances if i.is_known()]
        self.unknown = [i for i in self.instances if not i.is_known()]

    def get_features(self, insts):
        return np.array([inst.get_trace_features() if self.use_trace else inst.get_static_featuers() for inst in insts])

    def run(self):


        if self.train_with_unknown:
            folds = split_folds(self.labeled, self.unknown)
        else: 
            folds = split_folds(self.labeled)


        all_metrics = []
        all_eval = []
        unk_labeled_true = 0
        unk_labeled_false = 0

        for fold, (train, test) in enumerate(folds, 1):
            print(f"\n=== Fold {fold} ===")

            X_train = self.get_features(train)
            y_train = np.array([1 if i.get_label() else 0 for i in train])
            X_test = self.get_features(test)

            scaler = StandardScaler()
            X_all = scaler.fit_transform(np.vstack((X_train, X_test)))
            X_train_scaled = X_all[:len(X_train)]
            X_test_scaled = X_all[len(X_train):]

            self.clusterer.fit(X_train_scaled)
            self.clusterer.label_clusters(y_train, self.clusterer.clusterer.labels_, self.labeler)

            cluster_ids, strengths = self.clusterer.predict(X_test_scaled)

            if self.use_fallback:
                self.fallback.fit(X_train_scaled, y_train, self.clusterer.clusterer.labels_, self.clusterer.cluster_labels)
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
        overall["fold"] = "overall"
        all_metrics.append(overall)

        print("\n=== RF Overall ===")
        print(f"Precision: {overall['precision']:.3f}, Recall: {overall['recall']:.3f}, F1: {overall['f1']:.3f}")
        print(f"TP: {overall['TP']} | FP: {overall['FP']} | TN: {overall['TN']} | FN: {overall['FN']}")


        write_results_to_csv(all_eval, f"{self.output_dir}/cluster_results.csv")
        write_metrics_to_csv(all_metrics, f"{self.output_dir}/fold_metrics.csv")



    def evaluate(self, test):
        y_true = []
        y_pred = []
        
        true = 0
        false = 0

        for inst in test:
            if inst.is_known():
                y_true.append(int(inst.get_label()))
                y_pred.append(int(inst.get_predicted_label()))
            else:
                pl = inst.get_predicted_label()
                if pl == True:
                    true += 1
                elif pl == False:
                    false += 1
            
        
        print("all: ", true+false)
        print("true: ", true)
        print("false: ", false)
        
        return (evaluate_fold(y_true, y_pred), true, false)