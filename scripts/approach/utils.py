import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
import numpy as np

def balance_labeled_data_with_smote(instances, features):
    """
    Uses SMOTE to balance labeled (non-unknown) data.
    
    Parameters:
        instances: List[Instance] — full list of training instances
        features: List[np.ndarray] — corresponding features (same order)
    
    Returns:
        balanced_instances: List[Instance] (synthetic copies created)
        balanced_features: np.ndarray
        balanced_labels: np.ndarray
    """
    from copy import deepcopy

    # Extract known labels and features
    known = [(inst, x) for inst, x in zip(instances, features) if inst.is_known() and not inst.is_unknown]
    if not known:
        raise ValueError("No known labeled data to balance.")

    inst_list, X = zip(*known)
    y = np.array([1 if inst.get_label() else 0 for inst in inst_list])

    smote = SMOTE(sampling_strategy=1.0, k_neighbors=3,random_state=42)
    X_resampled, y_resampled = smote.fit_resample(np.array(X), y)

    # Reconstruct new instances
    from collections import defaultdict
    template_by_label = defaultdict(lambda: inst_list[0])
    for inst in inst_list:
        template_by_label[int(inst.get_label())] = inst

    balanced_instances = []
    for x, label in zip(X_resampled, y_resampled):
        new_inst = deepcopy(template_by_label[label])
        new_inst.set_static_features(x)
        new_inst.label = bool(label)
        new_inst.is_unknown = False
        balanced_instances.append(new_inst)

    return balanced_instances, X_resampled, y_resampled


def balance_training_set(train_instances):
    from sklearn.utils import resample

    true = [inst for inst in train_instances if inst.get_label() is True]
    false = [inst for inst in train_instances if inst.get_label() is False]

    print("before balance:")
    print(len(true))
    print(len(false))

    if len(true) > len(false):
        true = resample(true, replace=False, n_samples=len(false), random_state=42)
    else:
        false = resample(false, replace=False, n_samples=len(true), random_state=42)

    print("after balance:")
    print(len(true))
    print(len(false))
    return true + false


def split_folds(labeled_instances, unknown_instances=None, train_with_unknown=True , n_splits=5):
    true_instances = [inst for inst in labeled_instances if inst.get_label() is True]
    false_instances = [inst for inst in labeled_instances if inst.get_label() is False]

    # if unknown_instances:
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
        unk_false_train_idx, unk_false_test_idx = unk_false_splits[i]

        train_instances = [true_instances[j] for j in true_train_idx] + \
                          [false_instances[j] for j in false_train_idx]

        test_instances = [true_instances[j] for j in true_test_idx] + \
                         [false_instances[j] for j in false_test_idx] + \
                         [unk_false_instances[j] for j in unk_false_test_idx]
        

        if train_with_unknown:
            train_instances += [unk_false_instances[j] for j in unk_false_train_idx]
        
        # test_instances += [unk_false_instances[j] for j in unk_false_test_idx]

        folds.append((train_instances, test_instances))

    return folds


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



def evaluate(test):
    y_true, y_pred, gt_y_true, gt_y_pred = [], [], [], []
    true, false = 0, 0

    for inst in test:
        pred = int(inst.get_predicted_label())
        if inst.is_known():
            y_true.append(int(inst.get_label()))
            y_pred.append(pred)
        else:
            true += pred
            false += (1 - pred)
            if inst.ground_truth is not None:
                gt_y_true.append(int(inst.ground_truth))
                gt_y_pred.append(pred)

    eval_main = evaluate_fold(y_true, y_pred)
    eval_gt = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}
    return eval_main, true, false, eval_gt
