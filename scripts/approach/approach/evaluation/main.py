import os

from approach.data_representation import Instance
from approach.utils import evaluate_fold
# from approach.evaluation.evaluations import evaluate

import pandas as pd
import pickle as pkl
from collections import defaultdict
import numpy as np
import concurrent.futures
import functools


INSTANCES_DIR = '/20TB/mohammad/xcorpus-total-recall/results'
OUTPUT_DIR = '/20TB/mohammad/xcorpus-total-recall/evaluation'
OUTPUT_DIR_2 = '/20TB/mohammad/xcorpus-total-recall/evaluation/cluster'
MANUAL_LABELS_DIR = '/20TB/mohammad/xcorpus-total-recall/manual_labeling2'

dcg_path = '/20TB/mohammad/xcorpus-total-recall/dynamic_cgs'
manual_dcg_path = '/20TB/mohammad/xcorpus-total-recall/manual_labeling2'


def clustering_final_evaluation():


    def aggregate(outdir):
        all_dfs = []
        for f in os.listdir(outdir):

            if 'overall_evaluation.csv' in f:
                continue

            f_path = os.path.join(outdir, f)
            
            
            df = pd.read_csv(f_path)
            all_dfs.append(df)

        # calculate the mean of precision, recall, f1 and sum of tp, tn, fp, fn, with 2 floating point precision
        all_dfs = pd.concat(all_dfs, ignore_index=True)
        mean_metrics = all_dfs[['precision', 'recall', 'f1', 'gt_precision', 'gt_recall', 'gt_f1', 'cg_precision', 'cg_recall', 'cg_f1', 'cg_gt_precision','cg_gt_recall','cg_gt_f1']].mean().round(2).to_dict()
        sum_metrics = all_dfs[['TP', 'TN', 'FP', 'FN', 'gt_TP', 'gt_TN', 'gt_FP', 'gt_FN']].sum().to_dict()

        overall_metrics = {**mean_metrics, **sum_metrics}
        overall_df = pd.DataFrame(overall_metrics, index=[0])
        output_file = os.path.join(outdir, 'overall_evaluation.csv')
        overall_df.to_csv(output_file, index=False)

    BASE_DIR = '/20TB/mohammad/xcorpus-total-recall/evaluation/clustering'
    TARGET_NAMES = {"codebert", "codet5", "struct"}
    for root, dirs, files in os.walk(BASE_DIR):
        for dirname in dirs:
            if dirname in TARGET_NAMES:
                dir_path = os.path.join(root, dirname)
                aggregate(dir_path)
        



def evaluate(instances, true_only=True):
    y_true = []
    y_pred = []
    
    true = 0
    false = 0

    # For GT-labeled unknowns
    gt_y_true = []
    gt_y_pred = []

    for inst in instances:
        pred = int(inst.get_cluster_label())

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

    # if true_only: randomly delete 50% of y_true and y_pred that is true.
    if true_only:
        true_indices = [i for i, label in enumerate(y_true) if label == 1]
        false_indices = [i for i, label in enumerate(y_true) if label == 0]
        np.random.shuffle(true_indices)
        true_indices = true_indices[:len(true_indices) // 2]
        y_true = [y_true[i] for i in true_indices + false_indices]
        y_pred = [y_pred[i] for i in true_indices + false_indices]

    # if true_false: randoly delete 50% of y_true and y_pred that is true and 50% of y_true and y_pred that is false.
    else:
        true_indices = [i for i, label in enumerate(y_true) if label == 1]
        false_indices = [i for i, label in enumerate(y_true) if label == 0]
        np.random.shuffle(true_indices)
        np.random.shuffle(false_indices)
        true_indices = true_indices[:len(true_indices) // 2]
        false_indices = false_indices[:len(false_indices) // 2]
        y_true = [y_true[i] for i in true_indices + false_indices]
        y_pred = [y_pred[i] for i in true_indices + false_indices]

    print("all: ", true+false)
    print("true: ", true)
    print("false: ", false)
    print("Manually labeled unknowns:", len(gt_y_true))

    eval_main = evaluate_fold(y_true, y_pred)
    eval_gt = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}    
    return eval_main, true, false, eval_gt



def evaluate_cluster(instances, manual_gt_map, output_file):
    
    # find the instances that are in manual labeled map
    for inst in instances:
        key = (inst.src, inst.offset, inst.target)
        if key in manual_gt_map:
            inst.set_ground_truth(manual_gt_map[key])

    only_true = None
    if 'only_true' in output_file:
        only_true = True
    elif 'true_false' in output_file:
        only_true = False 

    res = evaluate(instances, only_true)
    cg_metrics = evaluate_callgraph(instances, is_cluster=True)


    metrics = res[0]
    metrics["unk_labeled_true"] = res[1]
    metrics["unk_labeled_false"] = res[2]
    metrics["unk_labeled_all"] = res[1] + res[2]

    gt_metrics = res[3]
    if gt_metrics:
        for k, v in gt_metrics.items():
            metrics[f"gt_{k}"] = v

    for k, v in cg_metrics.items():
        metrics[f'cg_{k}'] = v
    

    # save the evaluation results
    overall_df = pd.DataFrame([metrics], index=[0])
    overall_df.to_csv(output_file, index=False)


def evaluate_callgraph(instances, is_cluster=False):
    """
    Evaluate the callgraph of the instances and add the results to the overall metrics.
    """

    def compute_metrics(pred_edges, truth_edges):
        intersection = pred_edges & truth_edges
        precision = len(intersection) / len(pred_edges) if pred_edges else 0.0
        recall = len(intersection) / len(truth_edges) if truth_edges else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1
    
    def calculate_cg_metrics(instances, known_dcg_df, manual_dcg_df):
        """
        Calculate the callgraph metrics for the instances.
        """
        known_pruned = []
        manual_pruned = []

        for inst in instances:
            if is_cluster:
                inst.set_predicted_label(inst.get_cluster_label())
            if inst.is_known():
                if inst.get_predicted_label() == 1:
                    known_pruned.append(inst)
            elif inst.ground_truth is not None:
                if inst.get_predicted_label() == 1:
                    manual_pruned.append(inst)
        
        # Convert pruned instances to edge sets with offset
        known_pruned_edges = {(inst.src, inst.offset, inst.target) for inst in known_pruned}
        manual_pruned_edges = {(inst.src, inst.offset, inst.target) for inst in manual_pruned}

        # Convert dynamic callgraph DataFrames to sets with offset
        known_dcg_edges = set(zip(known_dcg_df['method'], known_dcg_df['offset'], known_dcg_df['target']))
        manual_dcg_edges = set(zip(manual_dcg_df['method'], manual_dcg_df['offset'], manual_dcg_df['target']))

        known_precision, known_recall, known_f1 = compute_metrics(known_pruned_edges, known_dcg_edges)
        manual_precision, manual_recall, manual_f1 = compute_metrics(manual_pruned_edges, manual_dcg_edges)

        return {
            'precision': known_precision,
            'recall': known_recall,
            'f1': known_f1,
            'gt_precision': manual_precision,
            'gt_recall': manual_recall,
            'gt_f1': manual_f1
        }   
        
            

    # seperate the instances by program
    program_map = defaultdict(list)
    for inst in instances:
        program_map[inst.program].append(inst)

    metrices = []
    for program, insts in program_map.items():
        
        # read the dynamic callgraph csv for program
        known_dcg_df = pd.read_csv(os.path.join(dcg_path, program, 'dcg_filtered.csv'))
        manual_dcg_df = pd.read_csv(os.path.join(manual_dcg_path, program, 'dcg.csv'))


        metrics = calculate_cg_metrics(insts, known_dcg_df, manual_dcg_df)
        metrices.append(metrics)

    # calculate the mean of the metrics
    overall_metrics = {}
    for key in metrices[0].keys():
        overall_metrics[key] = sum(m[key] for m in metrices) / len(metrices)
    
    return overall_metrics

# def evaluate_callgraph(instances, is_cluster=False):
#     """
#     Evaluates the callgraph by aggregating all predictions and ground truths
#     to compute a single set of metrics.

#     Args:
#         instances (list): A list of all instance objects across all programs.
#         dcg_path (str): The path to the directory containing known dynamic callgraph data.
#         manual_dcg_path (str): The path to the directory containing manual dynamic callgraph data.
#         is_cluster (bool): If True, use the cluster label as the predicted label.

#     Returns:
#         dict: A dictionary with the overall precision, recall, and F1-score.
#     """
    
#     def compute_metrics(pred_edges, truth_edges):
#         """Computes precision, recall, and F1-score from prediction and truth sets."""
#         intersection = pred_edges.intersection(truth_edges)
        
#         precision = len(intersection) / len(pred_edges) if pred_edges else 0.0
#         recall = len(intersection) / len(truth_edges) if truth_edges else 0.0
#         f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
#         return precision, recall, f1

#     known_pruned_edges = set()
#     manual_pruned_edges = set()

#     # Aggregate all predicted edges
#     for inst in instances:
#         if is_cluster:
#             inst.set_predicted_label(inst.get_cluster_label())
        
#         if inst.get_predicted_label() == 1:
#             edge = (inst.src, inst.offset, inst.target)
#             if inst.is_known():
#                 known_pruned_edges.add(edge)
#             elif inst.ground_truth is not None:
#                 manual_pruned_edges.add(edge)

#     # Determine unique programs to load ground truth data
#     programs = {inst.program for inst in instances}
    
#     all_known_dcg_edges = set()
#     all_manual_dcg_edges = set()

#     # Aggregate all ground truth edges
#     for program in programs:
#         known_dcg_file = os.path.join(dcg_path, program, 'dcg_filtered.csv')
#         if os.path.exists(known_dcg_file):
#             known_dcg_df = pd.read_csv(known_dcg_file)
#             all_known_dcg_edges.update(zip(known_dcg_df['method'], known_dcg_df['offset'], known_dcg_df['target']))

#         manual_dcg_file = os.path.join(manual_dcg_path, program, 'dcg.csv')
#         if os.path.exists(manual_dcg_file):
#             manual_dcg_df = pd.read_csv(manual_dcg_file)
#             all_manual_dcg_edges.update(zip(manual_dcg_df['method'], manual_dcg_df['offset'], manual_dcg_df['target']))

#     # Compute metrics on the globally aggregated sets
#     known_precision, known_recall, known_f1 = compute_metrics(known_pruned_edges, all_known_dcg_edges)
#     manual_precision, manual_recall, manual_f1 = compute_metrics(manual_pruned_edges, all_manual_dcg_edges)

#     return {
#         'precision': known_precision,
#         'recall': known_recall,
#         'f1': known_f1,
#         'gt_precision': manual_precision,
#         'gt_recall': manual_recall,
#         'gt_f1': manual_f1
#     }
        


def evaluate_runner(instances, manual_gt_map, output_file):
        # find the instances that are in manual labeled map
    for inst in instances:
        key = (inst.src, inst.offset, inst.target)
        if key in manual_gt_map:
            inst.set_ground_truth(manual_gt_map[key])
    

    # labeled evaluation
    labeled = [i for i in instances if i.is_known()]

    y_all_true = [1 if inst.get_label() else 0 for inst in labeled]
    y_all_pred = [1 if inst.get_predicted_label() else 0 for inst in labeled]

    overall = evaluate_fold(y_all_true, y_all_pred)

    # manual labeled evaluation
    gt_instances = [i for i in instances if i.ground_truth is not None and not i.is_known()]
    gt_y_true = [int(i.ground_truth) for i in gt_instances]
    gt_y_pred = [int(i.get_predicted_label()) for i in gt_instances]
    gt_metrics = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}
    for k, v in gt_metrics.items():
        overall[f"gt_{k}"] = v

    cg_metrics = evaluate_callgraph(instances, is_cluster=False)
    for k, v in cg_metrics.items():
        overall[f'cg_{k}'] = v

    # save the evaluation results
    overall_df = pd.DataFrame(overall, index=[0])
    overall_df.to_csv(output_file, index=False)


def load_manual_labels():

    dfs = []
    for program in os.listdir(MANUAL_LABELS_DIR):
        if not os.path.isdir(os.path.join(MANUAL_LABELS_DIR, program)):
            continue
        df = pd.read_csv(os.path.join(MANUAL_LABELS_DIR, program, 'complete', 'labeling_sample.csv'))
        dfs.append(df)

    manual_gt_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['method', 'offset', 'target'], keep='first')

    return manual_gt_df



# def main():


#     # read the manual labels df
#     manual_gt_map = {}
#     manual_gt_df = load_manual_labels()

#     for i, row in manual_gt_df.iterrows():
#         key = (row['method'], row['offset'], row['target'])
#         # if label is not int or is NaN, skip it
#         if pd.isna(row['label']):
#             continue
#         manual_gt_map[key] = int(row['label'])  # 0 or 1

#     # Loop through all subdirectories and files
#     for root, dirs, files in os.walk(INSTANCES_DIR):
#         for file in files:
#             if file.endswith('.pkl'):
#                 file_path = os.path.join(root, file)
#                 try:
#                     # print(f"Evaluating: {file_path}")
#                     # instances = pd.read_pickle(file_path)

#                     # if the instance is in "baseline" parent directory, evaluate it as a runner
#                     if 'baseline' in root:
#                         print(f"Evaluating: {file_path}")
#                         instances = pd.read_pickle(file_path)
#                         output_directory = os.path.join(OUTPUT_DIR, 'baseline', os.path.relpath(file_path, INSTANCES_DIR))
#                         os.makedirs(os.path.dirname(output_directory), exist_ok=True)
                        
#                         output_file = output_directory.replace('.pkl', '_evaluation.csv')
#                         print(f"Output file: {output_file}")
#                         evaluate_runner(instances, manual_gt_map, output_file)



#                     # if the instance is in "clustering" parent directory, evaluate it as a cluster
#                     elif 'clustering' in root:
#                         continue
#                         # print(f"Evaluating: {file_path}")
#                         # instances = pd.read_pickle(file_path)
#                         output_directory = os.path.join(OUTPUT_DIR_2, 'clustering', os.path.relpath(file_path, INSTANCES_DIR))
#                         os.makedirs(os.path.dirname(output_directory), exist_ok=True)

#                         output_file = output_directory.replace('.pkl', '_evaluation.csv')
#                         print(f"Output file: {output_file}")
#                         evaluate_cluster(instances, manual_gt_map, output_file)
#                     else:
#                         print(f"Unknown directory structure for {file_path}. Skipping evaluation.")

#                 except Exception as e:
#                     print(f"Failed to load {file_path}: {e}")

    
def process_file(file_path, manual_gt_map):
    """
    This function contains the logic to process one .pkl file.
    It will be executed by each thread in the pool.
    """
    try:
        print(f"Processing: {file_path}")
        instances = pd.read_pickle(file_path)

        # Construct the output path
        output_directory = os.path.join(OUTPUT_DIR, 'baseline', os.path.relpath(os.path.dirname(file_path), INSTANCES_DIR))
        os.makedirs(output_directory, exist_ok=True)
        
        output_file = os.path.join(output_directory, os.path.basename(file_path).replace('.pkl', '_evaluation.csv'))

        # Call the evaluation function
        evaluate_runner(instances, manual_gt_map, output_file)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # --- Setup (runs once) ---
    manual_gt_map = {}
    manual_gt_df = load_manual_labels()
    for _, row in manual_gt_df.iterrows():
        if not pd.isna(row['label']):
            key = (row['method'], row['offset'], row['target'])
            manual_gt_map[key] = int(row['label'])

    # 2. Collect all file paths first
    files_to_process = []
    for root, _, files in os.walk(INSTANCES_DIR):
        if 'baseline' in root:
            for file in files:
                # if file already exists in the output directory, skip it
                if file.endswith('.pkl'):
                    output_directory = os.path.join(OUTPUT_DIR, 'baseline', os.path.relpath(root, INSTANCES_DIR))
                    os.makedirs(output_directory, exist_ok=True)
                    output_file = os.path.join(output_directory, file.replace('.pkl', '_evaluation.csv'))
                    if not os.path.exists(output_file):
                        files_to_process.append(os.path.join(root, file))
    
    print(f"Found {len(files_to_process)} files to process.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:

        process_func = functools.partial(process_file, manual_gt_map=manual_gt_map)
        
        # The map function distributes the files across the threads
        list(executor.map(process_func, files_to_process)) 

    print("All files have been processed.")

    


if __name__ == "__main__":

    main()
    # clustering_final_evaluation()