import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from approach.utils import evaluate_fold, write_results_to_csv, write_metrics_to_csv, split_folds, balance_training_set, split_folds_programs

class RandomForestBaseline:

    def __init__(self, instances, output_dir, train_with_unknown=True, make_balance=False, threshold=0.5, raw_baseline=False, use_trace=False):
        self.instances = instances
        self.use_trace = use_trace
        self.raw_baseline = raw_baseline
        self.output_dir = output_dir
        self.threshold = threshold
        self.make_balance = make_balance
        self.train_with_unknown = train_with_unknown
        self.labeled = [i for i in instances if i.is_known()]
        self.unknown = [i for i in instances if not i.is_known()]

    def get_features(self, insts):
        return np.array([inst.get_static_featuers() for inst in insts])


    def run(self):

        # folds = split_folds(self.labeled, self.unknown, self.train_with_unknown)
        folds = split_folds_programs(self.instances, self.train_with_unknown)
        all_metrics = []
        all_eval = []
        unk_labeled_true = 0
        unk_labeled_false = 0

        for fold, (train, test) in enumerate(folds, 1):
            print(f"\n=== RF Fold {fold} ===")
            
            if self.make_balance:
                if self.make_balance[0] == "smote":
                    return
                else:
                    train = balance_training_set(train, self.make_balance[0], self.make_balance[1])

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
                criterion="entropy",
                class_weight='balanced'
            )
            clf.fit(X_train_scaled, y_train)

            # y_pred = clf.predict(X_test_scaled)
            # y_conf = clf.predict_proba(X_test_scaled).max(axis=1)
            probs = clf.predict_proba(X_test_scaled)[:, 1]  # Probabilities of class=1
            y_pred = (probs >= self.threshold).astype(int)
            y_conf = probs  # Confidence = p(class=1)


            for inst, pred, conf in zip(test, y_pred, y_conf):
                inst.set_predicted_label(pred)
                inst.set_confidence(conf)

            # metrics = evaluate_fold(y_test, y_pred)
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

        # Add evaluation on manually labeled unknowns
        gt_instances = [i for i in self.instances if i.ground_truth is not None and not i.is_known()]
        gt_y_true = [int(i.ground_truth) for i in gt_instances]
        gt_y_pred = [int(i.get_predicted_label()) for i in gt_instances]
        gt_metrics = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}
        for k, v in gt_metrics.items():
            overall[f"gt_{k}"] = v

        # calculate the mean of all the gt metrics
        # for metric in all_metrics:
        #     for k, v in metric.items():
        #         if k.startswith("gt_"):
        #             if k not in overall:
        #                 overall[k] = 0
        #             overall[k] += v
        # for k in overall.keys():
        #     if k.startswith("gt_"):
        #         overall[k] =  round(overall[k]/len(all_metrics),3)

        overall["fold"] = "overall"
        all_metrics.append(overall)

        print("\n=== RF Overall ===")
        print(f"Precision: {overall['precision']:.3f}, Recall: {overall['recall']:.3f}, F1: {overall['f1']:.3f}")
        print(f"TP: {overall['TP']} | FP: {overall['FP']} | TN: {overall['TN']} | FN: {overall['FN']}")

        if not self.raw_baseline:
            if self.train_with_unknown:
                metrics_path = f"{self.output_dir}/rf_{self.threshold}_trained_on_unknown.csv"
            elif self.make_balance:
                metrics_path = f"{self.output_dir}/rf_{self.threshold}_trained_on_known_{self.make_balance[0]}_{self.make_balance[1]}.csv"
            else:
                metrics_path = f"{self.output_dir}/rf_{self.threshold}_trained_on_known.csv"
        else:
            metrics_path = f"{self.output_dir}/rf_raw_{self.threshold}.csv"

        # write_results_to_csv(all_eval, res_path)
        write_metrics_to_csv(all_metrics, metrics_path)
        


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

                if inst.ground_truth is not None:
                    gt_y_true.append(int(inst.ground_truth))
                    gt_y_pred.append(pred)
                else:
                    if pred == 1:
                        true += 1
                    elif pred == 0:
                        false += 1

            
        
        print("all: ", true+false)
        print("true: ", true)
        print("false: ", false)
        print("Manually labeled unknowns:", len(gt_y_true))

        eval_main = evaluate_fold(y_true, y_pred)
        eval_gt = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}

        print(eval_gt)
        
        return eval_main, true, false, eval_gt
    

