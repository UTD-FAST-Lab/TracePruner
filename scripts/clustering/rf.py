from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

def rf_classifier_with_unknown_false(instances, use_trace=False, output_csv_path="rf_results.csv"):
    labeled = [inst for inst in instances if inst.is_known() and not inst.is_unknown]
    unknown = [inst for inst in instances if inst.is_unknown]

    def get_features(insts):
        return np.array([inst.get_trace_features() if use_trace else inst.get_static_featuers()
                         for inst in insts])

    def balance_training_set(train_instances):
        true = [inst for inst in train_instances if inst.get_label() is True]
        false = [inst for inst in train_instances if inst.get_label() is False]
        if len(true) > len(false):
            true = resample(true, replace=False, n_samples=len(false), random_state=42)
        else:
            false = resample(false, replace=False, n_samples=len(true), random_state=42)
        return true + false

    folds = split_per_class(labeled, n_splits=5)
    all_metrics = []
    all_eval_instances = []

    for fold, (train_instances, test_instances) in enumerate(folds, 1):
        print(f"\n=== RF Fold {fold} ===")

        train_instances = balance_training_set(train_instances)
        train_plus_unknown = train_instances + unknown

        X_train = get_features(train_plus_unknown)
        y_train = np.array([1 if inst.get_label() else 0 for inst in train_instances] +
                           [0 for _ in unknown])  # Unknowns treated as False

        X_test = get_features(test_instances)
        y_test = np.array([1 if inst.get_label() else 0 for inst in test_instances])

        # Normalize features
        scaler = StandardScaler()
        X_all = np.vstack((X_train, X_test))
        X_scaled = scaler.fit_transform(X_all)
        X_train_scaled = X_scaled[:len(X_train)]
        X_test_scaled = X_scaled[len(X_train):]

        # Train RF
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

        # Predict
        y_pred = clf.predict(X_test_scaled)
        y_conf = clf.predict_proba(X_test_scaled).max(axis=1)

        record_instance_prediction(test_instances, y_pred, y_conf)

        # Evaluation
        fold_metrics = evaluate_fold(y_test, y_pred)
        fold_metrics["fold"] = fold
        all_metrics.append(fold_metrics)
        all_eval_instances.extend(test_instances)

        print(f"RF Fold {fold} - F1: {fold_metrics['f1']:.3f}, Precision: {fold_metrics['precision']:.3f}, Recall: {fold_metrics['recall']:.3f}")
        print(f"TP: {fold_metrics['TP']} | FP: {fold_metrics['FP']} | TN: {fold_metrics['TN']} | FN: {fold_metrics['FN']}")

    # Overall eval
    y_all_true = [1 if inst.get_label() else 0 for inst in labeled if inst.get_predicted_label() is not None]
    y_all_pred = [1 if inst.get_predicted_label() else 0 for inst in labeled if inst.get_predicted_label() is not None]
    overall_metrics = evaluate_fold(y_all_true, y_all_pred)
    overall_metrics["fold"] = "overall"
    all_metrics.append(overall_metrics)

    print("\n=== RF Overall ===")
    print(f"Precision: {overall_metrics['precision']:.3f}")
    print(f"Recall:    {overall_metrics['recall']:.3f}")
    print(f"F1 Score:  {overall_metrics['f1']:.3f}")
    print(f"TP: {overall_metrics['TP']}, FP: {overall_metrics['FP']}, TN: {overall_metrics['TN']}, FN: {overall_metrics['FN']}")

    write_results_to_csv(all_eval_instances, output_csv_path)
    write_metrics_to_csv(all_metrics, "rf_fold_metrics.csv")
    print(f"\nRF results saved to {output_csv_path}")
    print("RF fold metrics saved to rf_fold_metrics.csv")
