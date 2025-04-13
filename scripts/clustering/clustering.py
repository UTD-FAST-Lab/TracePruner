import numpy as np
import pandas as pd
# from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import hdbscan
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter, defaultdict
import os

from Instance import Instance

def evaluate_fold(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "total_true": sum(y_true),
        "total_false": len(y_true) - sum(y_true)
    }

def write_results_to_csv(instances, path):
    records = []
    for inst in instances:
        records.append({
            "program": inst.program,
            "edge_id": inst.edge_id,
            "src": inst.src,
            "target": inst.target,
            "offset": inst.offset,
            "true_label": inst.label,
            "predicted_label": inst.predicted_label,
            "confidence": round(inst.confidence, 4) if inst.confidence is not None else None
        })
    pd.DataFrame(records).to_csv(path, index=False)

def write_metrics_to_csv(metrics, path):
    pd.DataFrame(metrics).to_csv(path, index=False)

def record_instance_prediction(instances, predictions, confidences):
    true = 0
    false = 0
    for inst, pred, conf in zip(instances, predictions, confidences):
        inst.set_predicted_label(bool(pred))
        inst.set_confidence(conf)
        # if inst.label == None:
        #     if bool(pred) == True:
        #         true += 1
        #     elif bool(pred) == False:
        #         false += 1
        if inst.is_unknown:
            if bool(pred) == True:
                true += 1
            elif bool(pred) == False:
                false += 1

    print(true)
    print(false)

def split_per_class(labeled_instances, unknown_instances=None, n_splits=5):
    from sklearn.model_selection import KFold
    true_instances = [inst for inst in labeled_instances if inst.get_label() is True]
    false_instances = [inst for inst in labeled_instances if inst.get_label() is False]

    if unknown_instances:
        unk_false_instances = [inst for inst in unknown_instances if inst.get_label() is False]
        unk_false_kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        unk_false_splits = list(unk_false_kf.split(unk_false_instances))


    true_kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    false_kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    true_splits = list(true_kf.split(true_instances))
    false_splits = list(false_kf.split(false_instances))

    folds = []
    for i in range(n_splits):
        true_train_idx, true_test_idx = true_splits[i]
        false_train_idx, false_test_idx = false_splits[i]

        train_instances = [true_instances[j] for j in true_train_idx] + \
                          [false_instances[j] for j in false_train_idx]

        test_instances = [true_instances[j] for j in true_test_idx] + \
                         [false_instances[j] for j in false_test_idx]
        
        if unknown_instances:
            unk_false_train_idx, unk_false_test_idx = unk_false_splits[i]

            train_instances += [unk_false_instances[j] for j in unk_false_train_idx]
            test_instances += [unk_false_instances[j] for j in unk_false_test_idx]

        folds.append((train_instances, test_instances))

    return folds

def balance_training_set(train_instances):
    from sklearn.utils import resample

    true = [inst for inst in train_instances if inst.get_label() is True]
    false = [inst for inst in train_instances if inst.get_label() is False]
    if len(true) > len(false):
        true = resample(true, replace=False, n_samples=len(false), random_state=42)
    else:
        false = resample(false, replace=False, n_samples=len(true), random_state=42)
    return true + false


def cluster_pipeline(instances, use_trace=False, output_csv_path="cluster_results.csv"):
    labeled = [inst for inst in instances if inst.is_known()]
    unlabeled = [inst for inst in instances if not inst.is_known()]

    def get_features(insts):
        return np.array([inst.get_trace_features() if use_trace else inst.get_static_featuers()
                         for inst in insts])

    folds = split_per_class(labeled, n_splits=5)

    all_metrics = []
    all_eval_instances = []

    for fold, (train_instances, test_instances) in enumerate(folds, 1):
        print(f"\n=== Fold {fold} ===")

        train_instances = balance_training_set(train_instances)


        X_train = get_features(train_instances)
        y_train = np.array([1 if inst.get_label() else 0 for inst in train_instances])
        X_test = get_features(test_instances)
        y_test = np.array([1 if inst.get_label() else 0 for inst in test_instances])
        X_unlabeled = get_features(unlabeled)

        # Normalize
        scaler = StandardScaler()
        X_all = np.vstack((X_train, X_test, X_unlabeled))
        X_all_scaled = scaler.fit_transform(X_all)
        X_train_scaled = X_all_scaled[:len(X_train)]
        X_test_scaled = X_all_scaled[len(X_train):len(X_train)+len(X_test)]
        X_unlabeled_scaled = X_all_scaled[len(X_train)+len(X_test):]

        # Fit HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
        clusterer.fit(X_train_scaled)

        # Majority label mapping
        cluster_to_labels = defaultdict(list)
        for label, cid in zip(y_train, clusterer.labels_):
            if cid != -1:
                cluster_to_labels[cid].append(label)
        cluster_label_map = {
            cid: Counter(labels).most_common(1)[0][0]
            for cid, labels in cluster_to_labels.items()
        }

        # Fallback KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)

        # Evaluate on (test + unlabeled)
        X_eval = np.vstack((X_test_scaled, X_unlabeled_scaled))
        eval_instances = test_instances + unlabeled
        cluster_ids, strengths = hdbscan.approximate_predict(clusterer, X_eval)
        knn_preds = knn.predict(X_eval)
        knn_probs = knn.predict_proba(X_eval).max(axis=1)

        predicted_classes = []
        predicted_confidences = []

        for i, (cid, strength) in enumerate(zip(cluster_ids, strengths)):
            if cid != -1 and cid in cluster_label_map:
                predicted_classes.append(cluster_label_map[cid])
                predicted_confidences.append(strength)
            else:
                predicted_classes.append(knn_preds[i])
                predicted_confidences.append(knn_probs[i])

        # Record results
        record_instance_prediction(eval_instances, predicted_classes, predicted_confidences)

        # Evaluation on labeled test portion only
        eval_test_preds = predicted_classes[:len(X_test)]
        fold_metrics = evaluate_fold(y_test, eval_test_preds)
        fold_metrics["fold"] = fold
        all_metrics.append(fold_metrics)

        if fold == 1:
            all_eval_instances.extend(eval_instances)

        print(f"Fold {fold} - F1: {fold_metrics['f1']:.3f}, Precision: {fold_metrics['precision']:.3f}, Recall: {fold_metrics['recall']:.3f}")
        print(f"TP: {fold_metrics['TP']} | FP: {fold_metrics['FP']} | TN: {fold_metrics['TN']} | FN: {fold_metrics['FN']}")

    # Overall eval
    y_all_true = [1 if inst.get_label() else 0 for inst in labeled if inst.get_predicted_label() is not None]
    y_all_pred = [1 if inst.get_predicted_label() else 0 for inst in labeled if inst.get_predicted_label() is not None]
    overall_metrics = evaluate_fold(y_all_true, y_all_pred)
    overall_metrics["fold"] = "overall"
    all_metrics.append(overall_metrics)

    print("\n=== Overall ===")
    print(f"Precision: {overall_metrics['precision']:.3f}")
    print(f"Recall:    {overall_metrics['recall']:.3f}")
    print(f"F1 Score:  {overall_metrics['f1']:.3f}")
    print(f"TP: {overall_metrics['TP']}, FP: {overall_metrics['FP']}, TN: {overall_metrics['TN']}, FN: {overall_metrics['FN']}")

    # Output
    write_results_to_csv(all_eval_instances, output_csv_path)
    write_metrics_to_csv(all_metrics, "fold_metrics.csv")
    print(f"\nResults saved to {output_csv_path}")
    print("Fold metrics saved to fold_metrics.csv")



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

    folds = split_per_class(labeled, unknown_instances= unknown , n_splits=5)
    all_metrics = []
    all_eval_instances = []

    for fold, (train_instances, test_instances) in enumerate(folds, 1):
        print(f"\n=== RF Fold {fold} ===")

        X_train = get_features(train_instances)
        y_train = np.array([1 if inst.get_label() else 0 for inst in train_instances])
                           

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



def load_instances():
    
    # programs_path = '/home/mohammad/projects/CallGraphPruner/scripts/trace-generation/programs.txt' 
    programs_path = '/home/mohammad/projects/CallGraphPruner/data/programs/all_programs.txt' 
    static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs'

    with open(programs_path, 'r') as f:
        program_names = [line.strip() for line in f if line.strip()]

    instances = []

    for program in program_names:

        program_dir_path = os.path.join(static_cg_dir, program)

        # File paths
        true_labels_path = os.path.join(program_dir_path, 'true_edges.csv')
        false_labels_path = os.path.join(program_dir_path, 'diff_0cfa_1obj.csv')
        all_edges_path = os.path.join(program_dir_path, 'wala0cfa.csv')
        static_features_path = os.path.join(program_dir_path, 'static_featuers', 'wala0cfa.csv')

        # Read CSVs safely
        if os.path.exists(true_labels_path):
            true_edges_df = pd.read_csv(true_labels_path)
        else:
            print(f"Missing true edges file for {program}")
            continue  # Or skip this program entirely

        if os.path.exists(false_labels_path):
            false_edges_df = pd.read_csv(false_labels_path)
        else:
            false_edges_df = pd.DataFrame(columns=['method', 'offset', 'target'])  # Empty if not found

        if os.path.exists(all_edges_path):
            all_edges_df = pd.read_csv(all_edges_path)
        else:
            print(f"Missing all edges file for {program}")
            continue

        if os.path.exists(static_features_path):
            static_features_df = pd.read_csv(static_features_path)
        else:
            print(f"Missing static features file for {program}")
            continue

        
        all_edges_df = all_edges_df[~all_edges_df['method'].str.startswith('java/') & ~all_edges_df['target'].str.startswith('java/')]


        # Combine true and false labels
        labeled_edges_df = pd.concat([true_edges_df, false_edges_df], ignore_index=True)

        # Keep only the relevant columns for comparison
        labeled_keys = labeled_edges_df[['method', 'offset', 'target']].drop_duplicates()

        # Use anti-join to find unknown edges
        unknown_edges_df = all_edges_df.merge(labeled_keys, on=['method', 'offset', 'target'], how='left', indicator=True)
        unknown_edges_df = unknown_edges_df[unknown_edges_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        # Optional: index static features for fast lookup
        static_features_df = static_features_df.set_index(['method', 'offset', 'target'])

        def create_instance(row, label=None, is_unknown = None):
            key = (row['method'], row['offset'], row['target'])
            instance = None
            # Attach static features if they exist
            if key in static_features_df.index:
                instance = Instance(program, *key, is_unknown, label=label)
                feature_row = static_features_df.loc[key]
                instance.set_static_features(feature_row)
            # else:
            #     # print(f"Warning: static features not found for edge: {key}")
            
            return instance

        for _, row in true_edges_df.iterrows():
            instance = create_instance(row, label=True, is_unknown = False)
            if not instance == None:
                instances.append(instance)

        for _, row in false_edges_df.iterrows():
            instance = create_instance(row, label=False, is_unknown = False)
            if not instance == None:
                instances.append(instance)

        for _, row in unknown_edges_df.iterrows():
            instance = create_instance(row, label=False, is_unknown = True)
            if not instance == None:
                instances.append(instance)

    return instances


if __name__ == '__main__':

    instances = load_instances()
    

    total_true = 0
    total_false = 0
    total_unk = 0
    for inst in instances:
        if inst.get_label() == True:
            total_true += 1
        elif inst.get_label() == False:
            total_false += 1

        if inst.is_unknown == True:
            total_unk += 1

    print(total_true, total_false, total_unk)

    # cluster_pipeline(instances, use_trace=False)

    rf_classifier_with_unknown_false(instances)