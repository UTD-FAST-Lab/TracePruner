import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from approach.utils import evaluate_fold, write_results_to_csv, write_metrics_to_csv, split_folds

class RandomForestBaseline:

    def __init__(self, instances, output_dir, raw_baseline=False, use_trace=False):
        self.instances = instances
        self.use_trace = use_trace
        self.raw_baseline = raw_baseline
        self.output_dir = output_dir
        self.labeled = [i for i in instances if i.is_known()]
        self.unknown = [i for i in instances if not i.is_known()]

    def get_features(self, insts):
        return np.array([inst.get_trace_features() if self.use_trace else inst.get_static_featuers() for inst in insts])


    def run(self):
        folds = split_folds(self.labeled, self.unknown)
        all_metrics = []
        all_eval = []
        unk_labeled_true = 0
        unk_labeled_false = 0

        for fold, (train, test) in enumerate(folds, 1):
            print(f"\n=== RF Fold {fold} ===")
            X_train = self.get_features(train)
            y_train = np.array([1 if i.get_label() else 0 for i in train])

            X_test = self.get_features(test)
            y_test = np.array([1 if i.get_label() else 0 for i in test])

            scaler = StandardScaler()
            X_all = np.vstack((X_train, X_test))
            X_scaled = scaler.fit_transform(X_all)
            X_train_scaled = X_scaled[:len(X_train)]
            X_test_scaled = X_scaled[len(X_train):]

            clf = RandomForestClassifier(
                n_estimators=1000,
                max_features="sqrt",
                random_state=0,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=False,
                criterion="entropy"
            )
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            y_conf = clf.predict_proba(X_test_scaled).max(axis=1)

            for inst, pred, conf in zip(test, y_pred, y_conf):
                inst.set_predicted_label(pred)
                inst.set_confidence(conf)

            # metrics = evaluate_fold(y_test, y_pred)
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

            print(f"RF Fold {fold} - F1: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
            print(f"TP: {metrics['TP']} | FP: {metrics['FP']} | TN: {metrics['TN']} | FN: {metrics['FN']}")

        # Overall
        if not self.raw_baseline:
            y_all_true = [1 if inst.get_label() else 0 for inst in self.labeled]
            y_all_pred = [1 if inst.get_predicted_label() else 0 for inst in self.labeled]
        else:
            y_all_true = [1 if inst.get_label() else 0 for inst in self.instances]
            y_all_pred = [1 if inst.get_predicted_label() else 0 for inst in self.instances]

        overall = evaluate_fold(y_all_true, y_all_pred)
        overall["unk_labeled_true"] = unk_labeled_true
        overall["unk_labeled_false"] = unk_labeled_false
        overall["unk_labeled_all"] = unk_labeled_false + unk_labeled_true
        overall["fold"] = "overall"
        all_metrics.append(overall)

        print("\n=== RF Overall ===")
        print(f"Precision: {overall['precision']:.3f}, Recall: {overall['recall']:.3f}, F1: {overall['f1']:.3f}")
        print(f"TP: {overall['TP']} | FP: {overall['FP']} | TN: {overall['TN']} | FN: {overall['FN']}")

        if not self.raw_baseline:
            res_path = f"{self.output_dir}/rf_results.csv"
            metrics_path = f"{self.output_dir}/rf_fold_metrics.csv"
        else:
            res_path = f"{self.output_dir}/rf_raw_results.csv"
            metrics_path = f"{self.output_dir}/rf_raw_fold_metrics.csv"

        write_results_to_csv(all_eval, res_path)
        write_metrics_to_csv(all_metrics, metrics_path)
        


    def evaluate(self, test):
        y_true = []
        y_pred = []
        
        true = 0
        false = 0

        for inst in test:

            if not self.raw_baseline:
                if inst.is_known():
                    y_true.append(int(inst.get_label()))
                    y_pred.append(int(inst.get_predicted_label()))
                else:
                    pl = inst.get_predicted_label()
                    if pl == True:
                        true += 1
                    elif pl == False:
                        false += 1
            else:
                y_true.append(int(inst.get_label()))
                y_pred.append(int(inst.get_predicted_label()))
                
        
        print("all: ", true+false)
        print("true: ", true)
        print("false: ", false)
        
        return (evaluate_fold(y_true, y_pred), true, false)
    

