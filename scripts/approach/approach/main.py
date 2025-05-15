import torch
from torch.utils.data import DataLoader
from approach.models.prediction_attn_model import ConfidenceWeightedNN, InstanceDataset, train_model, evaluate_model
from approach.data_representation.instance_loader import load_instances
from approach.clustering.flat_clustering_runner import FlatClusteringRunner
from approach.models.hdbscan_model import HDBSCANClusterer
from approach.utils import split_folds_instances, evaluate_fold, write_metrics_to_csv, custom_collate, scale_features




def run_clustering(instances, fold_idx):
    print(f"Running clustering for fold {fold_idx}...")

    runner = FlatClusteringRunner(
        instances=instances,
        clusterer=HDBSCANClusterer(min_cluster_size=5),
        only_true=True,
        output_dir="results/hdbscan/flat/trace/any_normal",
        use_trace=False,
        use_semantic=False,
        use_static=True,
        run_from_main=False,
    )

    # Run clustering
    runner.run()
    # Assuming the clustering algorithm modifies the instances in place


def main(instances, num_folds=5, feature_type='static', input_dim=11, hidden_dim=8, num_layers=2, dropout=0.3, batch_size=32, num_epochs=10, lr=1e-4, device='cuda'):

    all_metrics = []
    all_eval = []
    unk_labeled_true = 0
    unk_labeled_false = 0

    folds = split_folds_instances(instances, n_splits=num_folds)
    for fold_idx, (train_instances, test_instances) in enumerate(folds, 1):
        print(f"Processing Fold {fold_idx}...")

        # Run clustering to get labels and confidence scores
        run_clustering(train_instances, fold_idx)

        # scale the features
        scale_features(train_instances + test_instances)

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
        train_model(model, train_loader, optimizer, num_epochs=num_epochs, device=device)

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
    print_overal_eval(instances, all_metrics, unk_labeled_true, unk_labeled_false)


def print_overal_eval(instances, all_metrics, unk_labeled_true, unk_labeled_false):
    # Overall

    labeled = [i for i in instances if i.is_known()]

    y_all_true = [1 if inst.get_label() else 0 for inst in labeled]
    y_all_pred = [1 if inst.get_predicted_label() else 0 for inst in labeled]
    overall = evaluate_fold(y_all_true, y_all_pred)
    overall["unk_labeled_true"] = unk_labeled_true
    overall["unk_labeled_false"] = unk_labeled_false
    overall["unk_labeled_all"] = unk_labeled_false + unk_labeled_true
    
    # Add evaluation on manually labeled unknowns
    # gt_instances = [i for i in instances if i.ground_truth is not None and not i.is_known()]
    # gt_y_true = [int(i.ground_truth) for i in gt_instances]
    # gt_y_pred = [int(i.get_predicted_label()) for i in gt_instances]
    # gt_metrics = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}

    # for k, v in gt_metrics.items():
    #     overall[f"gt_{k}"] = v

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

    # # overall["gt_count"] = len(gt_y_true)
    # print(" ==== manual labeling ====")
    # print(gt_metrics)
    
    overall["fold"] = "overall"
    all_metrics.append(overall)

    print("\n=== RF Overall ===")
    print(f"Precision: {overall['precision']:.3f}, Recall: {overall['recall']:.3f}, F1: {overall['f1']:.3f}")
    print(f"TP: {overall['TP']} | FP: {overall['FP']} | TN: {overall['TN']} | FN: {overall['FN']}")

    write_metrics_to_csv(all_metrics, f"approach/results/prediction/struct-struct-attn.csv")


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
            if pred == 1:
                true += 1
            elif pred == 0:
                false += 1

            if inst.ground_truth is not None:
                gt_y_true.append(int(inst.ground_truth))
                gt_y_pred.append(pred)
        
    
    print("all: ", true+false)
    print("true: ", true)
    print("false: ", false)
    print("Manually labeled unknowns:", len(gt_y_true))

    eval_main = evaluate_fold(y_true, y_pred)
    eval_gt = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}

    print(eval_gt)
    
    return eval_main, true, false, eval_gt





if __name__ == '__main__':
    instances = load_instances('njr')
    print(f"Total instances: {len(instances)}")
    main(instances, device='cuda' if torch.cuda.is_available() else 'cpu')

