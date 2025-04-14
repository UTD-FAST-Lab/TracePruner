import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold


def split_folds(labeled_instances, unknown_instances=None, n_splits=5):
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
