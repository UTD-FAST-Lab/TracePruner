import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns


def scale_features(instances, feature_type='static'):
    features = []

    for instance in instances:
        if feature_type == 'static':
            feat = instance.get_static_featuers()
        elif feature_type == 'trace':
            feat = instance.get_trace_features()
        elif feature_type == 'semantic':
            feat = instance.get_semantic_features()
        elif feature_type == 'var':
            feat = instance.get_var_features()
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        if feat is not None:
            features.append(feat)

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Set the scaled features back to the instances
    for instance, scaled_feat in zip(instances, scaled_features):
        if feature_type == 'static':
            instance.set_static_features(scaled_feat)
        elif feature_type == 'trace':
            instance.set_trace_features(scaled_feat)
        elif feature_type == 'semantic':
            instance.set_semantic_features(scaled_feat)
        elif feature_type == 'var':
            instance.set_var_features(scaled_feat)

   



def custom_collate(batch):
    features, labels, confidences, instances = zip(*batch)
    # Stack features, labels, and confidences to form tensors
    return (
        torch.stack(features),
        torch.stack(labels),
        torch.stack(confidences),
        instances  # keep instances as a tuple for direct access
    )

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
    known = [(inst, x) for inst, x in zip(instances, features) if inst.is_known()]
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


def balance_training_set(train_instances, method="undersample", ratio=1.0, random_state=42):
    true = [inst for inst in train_instances if inst.get_label() is True]
    false = [inst for inst in train_instances if inst.get_label() is False]

    print("Before balance:")
    print(f"True: {len(true)}")
    print(f"False: {len(false)}")

    # Determine which is the majority and minority class
    if len(true) > len(false):
        majority = true
        minority = false
    else:
        majority = false
        minority = true


    if method == "undersample":
        # Calculate the target size based on the ratio
        target_size = int(len(minority) / ratio)
        target_size = min(target_size, len(majority))  # Ensure we don't exceed majority size
        majority_resampled = resample(majority, replace=False, n_samples=target_size, random_state=random_state)
        balanced_instances = majority_resampled + minority

    elif method == "oversample":
        # Oversample the minority class to the target size
        # Calculate the target size based on the ratio
        target_size = int(len(majority) * ratio)
        target_size = max(target_size, len(minority))  # Ensure we don't undersample minority
        minority_resampled = resample(minority, replace=True, n_samples=target_size, random_state=random_state)
        balanced_instances = majority + minority_resampled

    else:
        raise ValueError(f"Unsupported balancing method: {method}")

    print("After balance:")
    print(f"True: {sum(1 for inst in balanced_instances if inst.get_label() is True)}")
    print(f"False: {sum(1 for inst in balanced_instances if inst.get_label() is False)}")

    return balanced_instances

def split_folds_instances(instances, n_splits=5):
    
    labeled_instances = [inst for inst in instances if inst.is_known()]
    unknown_instances = [inst for inst in instances if not inst.is_known() and inst.ground_truth is None]
    manual_unknown_instances = [inst for inst in instances if not inst.is_known() and inst.ground_truth is not None]

    true_instances = [inst for inst in labeled_instances if inst.get_label() is True]
    false_instances = [inst for inst in labeled_instances if inst.get_label() is False]

    true_kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    false_kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    unk_kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)      

    true_splits = list(true_kf.split(true_instances))
    false_splits = list(false_kf.split(false_instances))
    unk_splits = list(unk_kf.split(unknown_instances))

    folds = []
    for i in range(n_splits):
        true_train_idx, true_test_idx = true_splits[i]
        false_train_idx, false_test_idx = false_splits[i]
        unk_train_idx, unk_test_idx = unk_splits[i]

        train_instances = [true_instances[j] for j in true_train_idx] + \
                          [false_instances[j] for j in false_train_idx] + \
                          [unknown_instances[j] for j in unk_train_idx]   

        test_instances = [true_instances[j] for j in true_test_idx] + \
                         [false_instances[j] for j in false_test_idx] + \
                         [unknown_instances[j] for j in unk_test_idx] + \
                         [manual_unknown_instances[j] for j in range(len(manual_unknown_instances))] 


        folds.append((train_instances, test_instances))

    return folds

def split_folds(labeled_instances, unknown_instances=None, train_with_unknown=True , n_splits=5):
    true_instances = [inst for inst in labeled_instances if inst.get_label() is True]
    false_instances = [inst for inst in labeled_instances if inst.get_label() is False]

    # if unknown_instances:
    unk_false_instances = [inst for inst in unknown_instances if inst.get_label() is False and inst.ground_truth is None]
    manual_unknown_instances = [inst for inst in unknown_instances if inst.ground_truth is not None]

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
                         [unk_false_instances[j] for j in unk_false_test_idx] +\
                         [manual_unknown_instances[j] for j in range(len(manual_unknown_instances))]
        

        if train_with_unknown:
            train_instances += [unk_false_instances[j] for j in unk_false_train_idx]
        
        # test_instances += [unk_false_instances[j] for j in unk_false_test_idx]

        folds.append((train_instances, test_instances))

    return folds


def split_folds_programs(instances, train_with_unknown=True, n_splits=5):
    ''' Splits instances into folds based on the programs they belong to'''

    from collections import defaultdict
    program_instances = defaultdict(list)
    for inst in instances:
        program_instances[inst.program].append(inst)
    program_list = list(program_instances.keys())
    print(program_list)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    program_splits = list(kf.split(program_list))
    folds = []
    for train_idx, test_idx in program_splits:
        train_programs = [program_list[i] for i in train_idx]
        test_programs = [program_list[i] for i in test_idx]

        train_instances = []
        test_instances = []

        for program in train_programs:
            if train_with_unknown:
                train_instances.extend(program_instances[program])
            else:
                # Only include known instances in training
                train_instances.extend([inst for inst in program_instances[program] if inst.is_known()])
        
        for program in test_programs:
            test_instances.extend(program_instances[program])
 
        folds.append((train_instances, test_instances))
    return folds


def split_fixed_set(instances, train_with_unknown=True):
    

    with open('/home/mohammad/projects/CallGraphPruner/data/programs/train_programs.txt', 'r') as f:
        training_programs = [line.strip() for line in f.readlines()]

    train_instances = []
    test_instances = []

    fold = []

    for inst in instances:
        if inst.program in training_programs:
            if train_with_unknown or inst.is_known():
                train_instances.append(inst)
        else:
            test_instances.append(inst)

    fold.append((train_instances, test_instances))

    return fold


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


def write_instances_to_file(instances, path):
    # write the instance to a file with pickle or something
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(instances, f)


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





import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_points(instances, path, feature_type='static', method='tsne', plot_only_known=False):
    """
    Plots instances in 2D using TSNE/PCA:
    - Green: known true
    - Red: known false
    - Gray (faint): unknown
    - If `plot_only_known=True`, skips unknowns entirely
    """

    feature_list = []
    colors = []
    sizes = []
    known_mask = []

    for inst in instances:
        if feature_type == 'static':
            vec = inst.get_static_featuers()
            expected_dim = 11
        elif feature_type == 'semantic':
            vec = inst.get_semantic_features()
            expected_dim = 768
        elif feature_type == 'trace':
            vec = inst.get_trace_features()
            expected_dim = 128
        else:
            raise ValueError("Invalid feature_type")

        if vec is None or len(vec) != expected_dim:
            print(f"Instance {inst} has invalid feature vector: {vec}")
            continue

        feature_list.append(vec)

        if inst.is_known():
            label = inst.get_label()
            colors.append('green' if label else 'red')
            sizes.append(5)
            known_mask.append(True)
        else:
            colors.append('black')
            sizes.append(2)
            known_mask.append(False)

    X = np.array(feature_list)
    known_mask = np.array(known_mask)

    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Invalid method: tsne or pca")

    X_embedded = reducer.fit_transform(X)

    # Split
    X_known = X_embedded[known_mask]
    known_colors = np.array(colors)[known_mask]
    known_sizes = np.array(sizes)[known_mask]

    plt.figure(figsize=(8, 6))

    if not plot_only_known:
        X_unknown = X_embedded[~known_mask]
        unknown_colors = np.array(colors)[~known_mask]
        unknown_sizes = np.array(sizes)[~known_mask]
        plt.scatter(X_unknown[:, 0], X_unknown[:, 1], c=unknown_colors, s=unknown_sizes, alpha=0.3, label='Unknown')

    plt.scatter(X_known[:, 0], X_known[:, 1], c=known_colors, s=known_sizes, alpha=0.6, label='Known')
    plt.title(f"{feature_type.upper()} Features ({method.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()





def plot_clusters_by_label(instances, features, cluster_ids, probs, save_path):
    pca = PCA(n_components=2)
    projection = pca.fit_transform(features)

    color_palette = sns.color_palette('hls', len(set(cluster_ids)))
    cluster_colors = [color_palette[x] if x >= 0 else (0.6, 0.6, 0.6) for x in cluster_ids]

    plt.figure(figsize=(10, 7))

    for i, inst in enumerate(instances):
        color = cluster_colors[i]
        marker = 'o' if inst.is_known() and inst.get_label() else \
                 'x' if inst.is_known() and not inst.get_label() else \
                 '.'
        plt.scatter(projection[i, 0], projection[i, 1],
                    c=[color], s=20,
                    marker=marker,
                    alpha=probs[i] if cluster_ids[i] != -1 else 0.3,
                    edgecolor='k' if marker != '.' else 'none', linewidth=0.2)

    plt.title("Cluster Visualization (shape=label, color=cluster)")
    plt.savefig(save_path)
    plt.close()
