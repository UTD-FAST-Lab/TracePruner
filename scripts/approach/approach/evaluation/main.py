import os

from approach.data_representation import Instance
from approach.utils import evaluate_fold
# from approach.evaluation.evaluations import evaluate

import pandas as pd
import pickle as pkl


def main():

    # read the instances file with picke format
    path = '/20TB/mohammad/xcorpus-total-recall/results/baseline/cgpruner/rf_programwise_0.45_trained_on_known_oversample_0.1.pkl'
    instances = pd.read_pickle(path)

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

    for k,v in manual_gt_map.items():
        print(f"Key: {k}, Value: {v}")

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

    # print the overall metrics
    print("Overall Metrics:")
    print(overall)
    


if __name__ == "__main__":

    main()