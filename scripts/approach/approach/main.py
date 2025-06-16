import torch
from torch.utils.data import DataLoader

import json, os
import itertools
from concurrent.futures import ThreadPoolExecutor

from approach.models.prediction_attn_model import ConfidenceWeightedNN, InstanceDataset, train_model, evaluate_model
from approach.data_representation.instance_loader import load_instances
from approach.clustering.flat_clustering_runner import FlatClusteringRunner
from approach.models.hdbscan_model import HDBSCANClusterer
from approach.utils import split_folds_instances, evaluate_fold, write_metrics_to_csv, custom_collate, scale_features, balance_training_set


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def run_clustering(instances, fold_idx, cluster_feature_type='static', only_true=True):
    print(f"Running clustering for fold {fold_idx}...")


    # Initialize the clustering algorithm
    if cluster_feature_type == 'static':
        options = load_config('approach/utils/best_config.json')['clustering']['struct']
        clusterer = HDBSCANClusterer(
            min_cluster_size=options['min_cluster_size'],
            min_samples=options['min_samples'],
            metric=options['metric'],
            cluster_selection_method=options['cluster_selection_method'],
            alpha=options['alpha'],
        )

        runner = FlatClusteringRunner(
            instances=instances,
            clusterer=clusterer,
            only_true=only_true,
            output_dir="",
            use_trace=False,
            use_semantic=False,
            use_static=True,
            run_from_main=False,
        )
    elif cluster_feature_type == 'trace':
        options = load_config('approach/utils/best_config.json')['clustering']['trace']
        clusterer = HDBSCANClusterer(
            min_cluster_size=options['min_cluster_size'],
            min_samples=options['min_samples'],
            metric=options['metric'],
            cluster_selection_method=options['cluster_selection_method'],
            alpha=options['alpha'],
        )

        runner = FlatClusteringRunner(
            instances=instances,
            clusterer=clusterer,
            only_true=only_true,
            output_dir="",
            use_trace=True,
            use_semantic=False,
            use_static=False,
            run_from_main=False,
        )
    elif cluster_feature_type == 'semantic':
        options = load_config('approach/utils/best_config.json')['clustering']['semantic_raw']
        clusterer = HDBSCANClusterer(
            min_cluster_size=options['min_cluster_size'],
            min_samples=options['min_samples'],
            metric=options['metric'],
            cluster_selection_method=options['cluster_selection_method'],
            alpha=options['alpha'],
        )

        runner = FlatClusteringRunner(
            instances=instances,
            clusterer=clusterer,
            only_true=only_true,
            output_dir="",
            use_trace=False,
            use_semantic=True,
            use_static=False,
            run_from_main=False,
        )

    # Run clustering
    runner.run()


def main(instances, num_folds=5, cluster_feature_type='static', only_true=True, feature_type='static', balance_type=None, input_dim=11, hidden_dim=8, num_layers=2, dropout=0.3, batch_size=32, num_epochs=10, lr=1e-4, apply_attention=True, device='cuda'):

    all_metrics = []
    all_eval = []
    unk_labeled_true = 0
    unk_labeled_false = 0

    folds = split_folds_instances(instances, n_splits=num_folds)
    for fold_idx, (train_instances, test_instances) in enumerate(folds, 1):
        print(f"Processing Fold {fold_idx}...")

        # Run clustering to get labels and confidence scores
        run_clustering(train_instances, fold_idx, cluster_feature_type=cluster_feature_type, only_true=only_true)

        # scale the features
        scale_features(train_instances + test_instances, feature_type=feature_type)

        # balance the dataset
        if balance_type is not None:
            train_instances = balance_training_set(train_instances, method=balance_type)

        # Prepare data loaders
        train_dataset = InstanceDataset(train_instances, feature_type=feature_type)
        test_dataset = InstanceDataset(test_instances, feature_type=feature_type)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

        # Initialize the model
        model = ConfidenceWeightedNN(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train the model
        print(f"Training Fold {fold_idx}...")
        train_model(model, train_loader, optimizer, num_epochs=num_epochs, device=device, apply_attention=apply_attention)

        # Evaluate the model
        print(f"Evaluating Fold {fold_idx}...")
        preds, labels = evaluate_model(model, test_loader, device=device)

        res = evaluate(test_instances)
        metrics = res[0]
        metrics["unk_labeled_true"] = res[1]
        metrics["unk_labeled_false"] = res[2]
        metrics["unk_labeled_all"] = res[1] + res[2]

        gt_metrics = res[3]
        if gt_metrics:
            for k, v in gt_metrics.items():
                metrics[f"gt_{k}"] = v

        metrics["fold"] = fold_idx

        unk_labeled_true += res[1]
        unk_labeled_false += res[2]
        all_metrics.append(metrics)
        all_eval.extend(test_instances)

        print(f"Fold {fold_idx } Evaluation Complete\n")


    # Overall evaluation
    print_overal_eval(instances, all_metrics, unk_labeled_true, unk_labeled_false, apply_attention=apply_attention, feature_type=feature_type, cluster_feature_type=cluster_feature_type, only_true=only_true, balance_type=balance_type)


def print_overal_eval(instances, all_metrics, unk_labeled_true, unk_labeled_false, apply_attention=True, feature_type='static', cluster_feature_type='static', only_true=True, balance_type=None):
    # Overall

    labeled = [i for i in instances if i.is_known()]

    y_all_true = [1 if inst.get_label() else 0 for inst in labeled]
    y_all_pred = [1 if inst.get_predicted_label() else 0 for inst in labeled]
    overall = evaluate_fold(y_all_true, y_all_pred)
    overall["unk_labeled_true"] = unk_labeled_true
    overall["unk_labeled_false"] = unk_labeled_false
    overall["unk_labeled_all"] = unk_labeled_false + unk_labeled_true

    # calculate the mean of all the gt metrics
    for metric in all_metrics:
        for k, v in metric.items():
            if k.startswith("gt_"):
                if k not in overall:
                    overall[k] = 0
                overall[k] += v
    for k in overall.keys():
        if k.startswith("gt_"):
            overall[k] =  round(overall[k]/len(all_metrics),3)

    
    overall["fold"] = "overall"
    all_metrics.append(overall)

    print("\n=== RF Overall ===")
    print(f"Precision: {overall['precision']:.3f}, Recall: {overall['recall']:.3f}, F1: {overall['f1']:.3f}")
    print(f"TP: {overall['TP']} | FP: {overall['FP']} | TN: {overall['TN']} | FN: {overall['FN']}")

    if only_true:
        if apply_attention:
            write_metrics_to_csv(all_metrics, f"approach/results/prediction/{cluster_feature_type}_only_true_{feature_type}_{balance_type}_attn.csv")
        else:
            write_metrics_to_csv(all_metrics, f"approach/results/prediction/{cluster_feature_type}_only_true_{feature_type}_{balance_type}.csv")
    else:
        if apply_attention:
            write_metrics_to_csv(all_metrics, f"approach/results/prediction/{cluster_feature_type}_true_false_{feature_type}_{balance_type}_attn.csv")
        else:
            write_metrics_to_csv(all_metrics, f"approach/results/prediction/{cluster_feature_type}_true_false_{feature_type}_{balance_type}.csv")

def evaluate(test):
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







def run_main_task(config, instances, cluster_feature_type, feature_type, only_true_val, balance_type, attention):
    # Extract the correct hyperparameters based on feature type
    if feature_type == 'static':
        params = config['struct']
    elif feature_type == 'trace':
        params = config['trace']
    elif feature_type == 'semantic':
        params = config['semantic_raw']
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    # Run the main function
    main(
        instances,
        num_layers=params['num_layers'],
        input_dim=params['input_dim'],
        hidden_dim=params['hidden_dim'],
        dropout=params['dropout'],
        feature_type=feature_type,
        num_epochs=params['num_epochs'],
        lr=params['lr'],
        batch_size=params['batch_size'],
        apply_attention=attention,
        cluster_feature_type=cluster_feature_type,
        only_true=only_true_val,
        balance_type=balance_type,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


def main_parallel(config, instances, clustering_feature_types, feature_types, only_true, balance_types, apply_attention, max_workers=8):
    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create all possible parameter combinations
        tasks = itertools.product(
            clustering_feature_types,
            feature_types,
            only_true,
            balance_types,
            apply_attention
        )

        # Submit each combination as a separate task
        futures = [
            executor.submit(
                run_main_task,
                config,
                instances,
                cluster_feature_type,
                feature_type,
                only_true_val,
                balance_type,
                attention
            )
            for cluster_feature_type, feature_type, only_true_val, balance_type, attention in tasks
        ]

        # Wait for all tasks to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")





if __name__ == '__main__':
    instances = load_instances('njr')
    print(f"Total instances: {len(instances)}")

    config = load_config('approach/utils/best_config.json')['prediction']

    feature_types = ['static', 'trace', 'semantic']
    clustering_feature_types = ['static', 'trace', 'semantic']
    only_true = [True, False]
    apply_attention = [True, False]
    balance_types = [None, 'undersample', 'oversample']

    main_parallel(
        config,
        instances,
        clustering_feature_types=clustering_feature_types,
        feature_types=feature_types,
        only_true=only_true,
        balance_types=balance_types,
        apply_attention=apply_attention,
        max_workers=8
    )

    # for cluster_feature_type in clustering_feature_types:
    #     for feature_type in feature_types:
    #         for only_true_val in only_true:
    #             for balance_type in balance_types:
    #                 for attention in apply_attention:

    #                     if feature_type == 'static':
    #                         input_dim = config['struct']['input_dim']
    #                         hidden_dim = config['struct']['hidden_dim']
    #                         num_layers = config['struct']['num_layers']
    #                         dropout = config['struct']['dropout']
    #                         batch_size = config['struct']['batch_size']
    #                         num_epochs = config['struct']['num_epochs']
    #                         lr = config['struct']['lr']
    #                     elif feature_type == 'trace':
    #                         input_dim = config['trace']['input_dim']
    #                         hidden_dim = config['trace']['hidden_dim']
    #                         num_layers = config['trace']['num_layers']
    #                         dropout = config['trace']['dropout']
    #                         batch_size = config['trace']['batch_size']
    #                         num_epochs = config['trace']['num_epochs']
    #                         lr = config['trace']['lr']
    #                     elif feature_type == 'semantic':
    #                         input_dim = config['semantic_raw']['input_dim']
    #                         hidden_dim = config['semantic_raw']['hidden_dim']
    #                         num_layers = config['semantic_raw']['num_layers']
    #                         dropout = config['semantic_raw']['dropout']
    #                         batch_size = config['semantic_raw']['batch_size']
    #                         num_epochs = config['semantic_raw']['num_epochs']
    #                         lr = config['semantic_raw']['lr']

    #                     main(
    #                         instances,
    #                         num_layers=num_layers,
    #                         input_dim=input_dim,
    #                         hidden_dim=hidden_dim,
    #                         dropout=dropout,
    #                         feature_type=feature_type,
    #                         num_epochs=num_epochs,
    #                         lr=lr,
    #                         batch_size=batch_size,
    #                         apply_attention=attention,
    #                         cluster_feature_type=cluster_feature_type,
    #                         only_true=only_true_val,
    #                         balance_type=balance_type,
    #                         device='cuda' if torch.cuda.is_available() else 'cpu'
    #                     )




