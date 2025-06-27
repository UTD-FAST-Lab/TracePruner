import os

from approach.data_representation import Instance
from approach.utils import evaluate_fold
# from approach.evaluation.evaluations import evaluate

import pandas as pd
import pickle as pkl


INSTANCES_DIR = '/20TB/mohammad/xcorpus-total-recall/results'
OUTPUT_DIR = '/20TB/mohammad/xcorpus-total-recall/evaluation'
OUTPUT_DIR_2 = '/20TB/mohammad/xcorpus-total-recall/evaluation/cluster'
MANUAL_LABELS_DIR = '/20TB/mohammad/xcorpus-total-recall/manual_labeling2'


def clustering_final_evaluation():


    def aggregate(outdir):
        all_dfs = []
        for f in os.listdir(outdir):
            f_path = os.path.join(outdir, f)
            
            
            df = pd.read_csv(f_path)
            all_dfs.append(df)

        # calculate the mean of precision, recall, f1 and sum of tp, tn, fp, fn
        all_dfs = pd.concat(all_dfs, ignore_index=True)
        mean_metrics = all_dfs[['precision', 'recall', 'f1', 'gt_precision', 'gt_recall', 'gt_f1']].mean().to_dict()
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
        



def evaluate(instances):
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
    

    res = evaluate(instances)
    metrics = res[0]
    metrics["unk_labeled_true"] = res[1]
    metrics["unk_labeled_false"] = res[2]
    metrics["unk_labeled_all"] = res[1] + res[2]

    gt_metrics = res[3]
    if gt_metrics:
        for k, v in gt_metrics.items():
            metrics[f"gt_{k}"] = v
    

    # save the evaluation results
    overall_df = pd.DataFrame([metrics], index=[0])
    overall_df.to_csv(output_file, index=False)



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

    # save the evaluation results
    overall_df = pd.DataFrame(overall, index=[0])
    overall_df.to_csv(output_file, index=False)



def main():


    # read the manual labels df
    manual_gt_map = {}
    manual_labels_path = '/20TB/mohammad/xcorpus-total-recall/manual_labeling/axion/complete/labeling_sample.csv'
    manual_gt_df = pd.read_csv(manual_labels_path)
    for i, row in manual_gt_df.iterrows():
        key = (row['method'], row['offset'], row['target'])
        # if label is not int or is NaN, skip it
        if pd.isna(row['label']):
            continue
        manual_gt_map[key] = int(row['label'])  # 0 or 1

    # Loop through all subdirectories and files
    for root, dirs, files in os.walk(INSTANCES_DIR):
        for file in files:
            if file.endswith('.pkl'):
                file_path = os.path.join(root, file)
                try:
                    print(f"Evaluating: {file_path}")
                    instances = pd.read_pickle(file_path)

                    # if the instance is in "baseline" parent directory, evaluate it as a runner
                    if 'baseline' in root:
                        # directory structure (same folder structure as the instance and file name but in output directory)
                        output_directory = os.path.join(OUTPUT_DIR, 'baseline', os.path.relpath(file_path, INSTANCES_DIR))
                        os.makedirs(os.path.dirname(output_directory), exist_ok=True)
                        
                        output_file = output_directory.replace('.pkl', '_evaluation.csv')
                        print(f"Output file: {output_file}")
                        evaluate_runner(instances, manual_gt_map, output_file)



                    # if the instance is in "clustering" parent directory, evaluate it as a cluster
                    elif 'clustering' in root:
                        output_directory = os.path.join(OUTPUT_DIR_2, 'clustering', os.path.relpath(file_path, INSTANCES_DIR))
                        os.makedirs(os.path.dirname(output_directory), exist_ok=True)

                        output_file = output_directory.replace('.pkl', '_evaluation.csv')
                        print(f"Output file: {output_file}")
                        evaluate_cluster(instances, manual_gt_map, output_file)
                    else:
                        print(f"Unknown directory structure for {file_path}. Skipping evaluation.")

                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

    


    


if __name__ == "__main__":

    main()