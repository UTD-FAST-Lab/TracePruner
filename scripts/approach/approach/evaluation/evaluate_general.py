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


INSTANCES_DIR = '/20TB/mohammad/xcorpus-total-recall/results2'
OUTPUT_DIR = '/20TB/mohammad/xcorpus-total-recall/evaluation2'
OUTPUT_DIR_2 = '/20TB/mohammad/xcorpus-total-recall/evaluation/cluster'
MANUAL_LABELS_DIR = '/20TB/mohammad/xcorpus-total-recall/manual_labeling2'

dcg_path = '/20TB/mohammad/xcorpus-total-recall/dynamic_cgs'
manual_dcg_path = '/20TB/mohammad/xcorpus-total-recall/manual_labeling2'


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


def evaluate_runner(instances, manual_gt_map, output_file):
        # find the instances that are in manual labeled map
    for inst in instances:
        key = (inst.src, inst.offset, inst.target)
        if key in manual_gt_map:
            inst.set_ground_truth(manual_gt_map[key])
    

    # seperate per configuration
    config_map = defaultdict(list)
    for inst in instances:
        config_map[(inst.tool, inst.version, inst.config_id)].append(inst)

    

    for config, instances in config_map.items():

        # print(f"Evaluating configuration: {config} with {len(instances)} instances")

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
        overall_df.to_csv(output_file.replace('.csv', f'_{config[0]}_{config[1]}_{config[2]}.csv'), index=False)


def load_manual_labels():

    dfs = []
    for program in os.listdir(MANUAL_LABELS_DIR):
        if not os.path.isdir(os.path.join(MANUAL_LABELS_DIR, program)):
            continue
        df = pd.read_csv(os.path.join(MANUAL_LABELS_DIR, program, 'complete', 'labeling_sample.csv'))
        dfs.append(df)

    manual_gt_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['method', 'offset', 'target'], keep='first')

    return manual_gt_df


    
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