import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from approach.utils import evaluate_fold, write_metrics_to_csv, split_folds_programs
from approach.data_representation.instance_loader import load_instances

class SVMBaseline:

    def __init__(self, instances, output_dir, kernel="rbf", nu=0.1, gamma="scale"):
        self.instances = instances
        self.output_dir = output_dir
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.labeled = [i for i in instances if i.is_known()]
        self.unknown = [i for i in instances if not i.is_known()]

    def get_features(self, insts):
        return np.array([inst.get_static_featuers() for inst in insts])

    def run(self):
        folds = split_folds_programs(self.instances, train_with_unknown=False)
        all_metrics = []
        all_eval = []
        unk_labeled_true = 0
        unk_labeled_false = 0

        for fold, (train, test) in enumerate(folds, 1):
            print(f"\n=== SVM Fold {fold} ===")

            # Only train on known TRUE instances
            pos_train = [i for i in train if i.get_label()]
            X_train = self.get_features(pos_train)

            X_test = self.get_features(test)
            y_test = np.array([1 if i.get_label() else 0 for i in test])

            # Normalize
            scaler = StandardScaler()
            X_all = np.vstack((X_train, X_test))
            X_scaled = scaler.fit_transform(X_all)
            X_train_scaled = X_scaled[:len(X_train)]
            X_test_scaled = X_scaled[len(X_train):]

            clf = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
            clf.fit(X_train_scaled)

            y_pred_ocsvm = clf.predict(X_test_scaled)
            y_pred = np.where(y_pred_ocsvm == 1, 1, 0)  # 1: in-class, -1: outliers → convert to binary
            y_conf = clf.decision_function(X_test_scaled)  # Higher = more confident in-class

            for inst, pred, conf in zip(test, y_pred, y_conf):
                inst.set_predicted_label(pred)
                inst.set_confidence(conf)

            res = self.evaluate(test)
            metrics = res[0]
            metrics["unk_labeled_true"] = res[1]
            metrics["unk_labeled_false"] = res[2]
            metrics["unk_labeled_all"] = res[1] + res[2]

            gt_metrics = res[3]
            if gt_metrics:
                for k, v in gt_metrics.items():
                    metrics[f"gt_{k}"] = v

            metrics["fold"] = fold

            unk_labeled_true += res[1]
            unk_labeled_false += res[2]
            all_metrics.append(metrics)
            all_eval.extend(test)

            print(f"SVM Fold {fold} - F1: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
            print(f"TP: {metrics['TP']} | FP: {metrics['FP']} | TN: {metrics['TN']} | FN: {metrics['FN']}")

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

        overall["fold"] = "overall"
        all_metrics.append(overall)

        print("\n=== SVM Overall ===")
        print(f"Precision: {overall['precision']:.3f}, Recall: {overall['recall']:.3f}, F1: {overall['f1']:.3f}")
        print(f"TP: {overall['TP']} | FP: {overall['FP']} | TN: {overall['TN']} | FN: {overall['FN']}")

        metrics_path = f"{self.output_dir}/svm_{self.kernel}_nu{self.nu}.csv"
        write_metrics_to_csv(all_metrics, metrics_path)

    def evaluate(self, test):
        y_true = []
        y_pred = []
        true = 0
        false = 0

        gt_y_true = []
        gt_y_pred = []

        for inst in test:
            pred = int(inst.get_predicted_label())

            if inst.is_known():
                y_true.append(int(inst.get_label()))
                y_pred.append(pred)
            else:
                if inst.ground_truth is not None:
                    gt_y_true.append(int(inst.ground_truth))
                    gt_y_pred.append(pred)
                else:
                    if pred == 1:
                        true += 1
                    else:
                        false += 1

        eval_main = evaluate_fold(y_true, y_pred)
        eval_gt = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}
        return eval_main, true, false, eval_gt



def main():

    instances = load_instances('njr')
    output_dir = "approach/results/svm"
    svm_runner = SVMBaseline(instances, output_dir, kernel="rbf", nu=0.1, gamma="scale")
    svm_runner.run()


if __name__ == "__main__":
    main()