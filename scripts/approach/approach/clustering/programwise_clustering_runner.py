
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.decomposition import FastICA

from approach.utils import evaluate_fold, write_metrics_to_csv, plot_clusters_by_label



class FlatClusteringRunner:
    def __init__(self, instances, clusterer, output_dir, run_from_main, only_true=True, use_trace=False, use_semantic=False, use_static=True, use_var=False, option=None):
        self.instances = instances
        self.clusterer = clusterer
        self.only_true = only_true
        self.run_from_main = run_from_main
        self.output_dir = output_dir
        self.use_trace = use_trace
        self.use_semantic = use_semantic
        self.use_static = use_static
        self.use_var = use_var
        self.option = option


        self.labeled = [i for i in instances if i.is_known()]
        self.unknown = [i for i in instances if not i.is_known()]

    def get_features(self, insts):
        feature_vecs = []

        # Collect feature vectors based on selected flags
        semantic_vecs = []
        trace_vecs = []
        var_vecs = []
        struct_vecs = []

        for inst in insts:
            # Semantic Features
            if self.use_semantic:
                code = inst.get_semantic_features()
                if code is None or len(code) != 768:
                    code = [0.0] * 768
                semantic_vecs.append(code)

            # Trace Features
            if self.use_trace:
                trace = inst.get_trace_features()
                # if trace is None or len(trace) != 64:
                #     trace = [0.0] * 64
                if trace is None or len(trace) != 128:
                    trace = [0.0] * 128
                trace_vecs.append(trace)

            # Variable Features
            if self.use_var:  
                var = inst.get_var_features()
                if var is None or len(var) != 20:
                    var = [0.0] * 20
                var_vecs.append(var)

            # Structural Features
            if self.use_static:
                struct = inst.get_static_featuers()
                if struct is None or len(struct) != 11:
                    struct = [0.0] * 11
                struct_vecs.append(struct)

        # Transform and scale features
        if len(var_vecs) > 0:
            var_vecs = StandardScaler().fit_transform(var_vecs)
        if len(struct_vecs) > 0:
            struct_vecs = StandardScaler().fit_transform(struct_vecs)
        if len(trace_vecs) > 0:
            trace_vecs = FastICA(n_components=min(50, len(trace_vecs[0]))).fit_transform(trace_vecs)
        if len(semantic_vecs) > 0:
            semantic_vecs = FastICA(n_components=min(50, len(semantic_vecs[0]))).fit_transform(semantic_vecs)

        # Combine features for each instance
        for i in range(len(insts)):
            combined = []

            if self.use_semantic:
                combined.extend(semantic_vecs[i])
            if self.use_trace:
                combined.extend(trace_vecs[i])
            if self.use_var:  
                combined.extend(var_vecs[i])
            if self.use_static:
                combined.extend(struct_vecs[i])

            feature_vecs.append(combined)

        return np.array(feature_vecs)

    def run(self):
        instances = self.instances
        X_scaled = self.get_features(instances)
        print("found %d instances" % len(X_scaled))

        # Split true and false labels for seeding
        true_instances = [i for i in self.labeled if i.get_label() is True]
        false_instances = [i for i in self.labeled if i.get_label() is False]

        # 50% seeding
        true_labeled, _ = train_test_split(true_instances, test_size=0.5, random_state=42)
        if len(false_instances) > 2:
            false_labeled, _ = train_test_split(false_instances, test_size=0.5, random_state=42)

        # Build y_seed based on labeling mode
        y_seed = np.full(len(instances), -1)
        seed_indices = []

        if self.only_true:
            seed_indices = [i for i, inst in enumerate(instances) if inst in true_labeled]
            for i in seed_indices:
                y_seed[i] = 1
        else:
            true_indices = [i for i, inst in enumerate(instances) if inst in true_labeled]
            false_indices = [i for i, inst in enumerate(instances) if inst in false_labeled]
            seed_indices = true_indices + false_indices
            for i in true_indices:
                y_seed[i] = 1
            for i in false_indices:
                y_seed[i] = 0

        # Fit clusterer
        self.clusterer.fit(X_scaled)

        # Get cluster labels
        cluster_ids = self.clusterer.clusterer.labels_
        cluster_labels = self.label_clusters(y_seed, cluster_ids)

        # self.print_cluster_distribution(instances, cluster_ids)


        # Apply fallback for outliers
        seed_X = X_scaled[seed_indices] if self.only_true else X_scaled[true_indices]
        predictions = []
        for i, cid in enumerate(cluster_ids):
            if cid != -1:
                pred = cluster_labels.get(cid, 0)  # default to 0 if cluster not in label map
                confidence = self.clusterer.clusterer.probabilities_[i]
            else:
                pred = self.fallback(seed_X, X_scaled[i])
                confidence = 0.0
            predictions.append(pred)
            instances[i].set_cluster_label(pred)
            instances[i].set_confidence(confidence)


        evaluated_instatnces = [inst for inst in instances if inst not in true_labeled]
        # remove instances that are in the true seeds
        
        res = self.evaluate(evaluated_instatnces)
        metrics = res[0]
        metrics["unk_labeled_true"] = res[1]
        metrics["unk_labeled_false"] = res[2]
        metrics["unk_labeled_all"] = res[1] + res[2]

        gt_metrics = res[3]
        if gt_metrics:
            for k, v in gt_metrics.items():
                metrics[f"gt_{k}"] = v

        print("\n=== Flat Clustering Evaluation ===")
        print(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
        print(f"TP: {metrics['TP']} | FP: {metrics['FP']} | TN: {metrics['TN']} | FN: {metrics['FN']}")


        if self.run_from_main:
            if self.only_true:
                labeler = "only_true"
            else:
                labeler = "true_false"

            if self.use_trace:
                write_metrics_to_csv([metrics], f'{self.output_dir}/{labeler}/trace_tuning/trace_{self.option}.csv' )
            elif self.use_semantic:
                write_metrics_to_csv([metrics], f'{self.output_dir}/{labeler}/semantic.csv' )
            elif self.use_static:
                program_output_dir = f'{self.output_dir}/{labeler}'
                os.makedirs(program_output_dir, exist_ok=True)
                write_metrics_to_csv([metrics], f'{program_output_dir}/struct.csv' )



    def fallback(self, seed_X, point):
        distances = cdist([point], seed_X)[0]
        return int(np.any(distances == 0.0))


    def label_clusters(self, y_seed, cluster_ids):
        cluster_to_labels = defaultdict(list)

        for idx, (cid, label) in enumerate(zip(cluster_ids, y_seed)):
            if cid != -1 and label != -1:
                cluster_to_labels[cid].append(label)

        # Apply labeling heuristic
        if self.only_true:
            # Only use true seed labels
            return {cid: 1 for cid, labels in cluster_to_labels.items() if 1 in labels}
        else:
            # Majority voting
            return {cid: 1 if sum(labels) > len(labels) // 2 else 0 for cid, labels in cluster_to_labels.items()}


    def evaluate(self, test):
        y_true = []
        y_pred = []
        
        true = 0
        false = 0

        # For GT-labeled unknowns
        gt_y_true = []
        gt_y_pred = []

        for inst in test:
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

        print(eval_gt)
        
        return eval_main, true, false, eval_gt


    def print_cluster_distribution(self, train, cluster_ids):
        '''print in each cluster how many true, false and unknowns are there'''

        cluster_2_data = defaultdict(list)

        for inst, cid in zip(train, cluster_ids):
            cluster_2_data[cid].append(inst)
        

        with open(f'{self.output_dir}/cluster_distibution.txt', 'w') as f:
            for cid, val in cluster_2_data.items():
                true = 0
                false = 0
                unk = 0
                for inst in val:
                    if not inst.is_known():
                        unk += 1
                    else:
                        if inst.get_label() is True:
                            true += 1
                        else:
                            false += 1
                
                f.write(f"cid: {cid} -> true: {true}\t false: {false}\t unk: {unk}\n ")



if __name__ == "__main__":

    outdir = "approach/results/clustering/programwise"
    all_dfs = []
    for program in os.listdir(outdir):
        program_dir = os.path.join(outdir, program, 'only_true', 'struct.csv')
        
        
        df = pd.read_csv(program_dir)
        all_dfs.append(df)
        # if (df['total_false'] > 0).any():
        #     print(f"Program {program} has false instances")
        #     os.system(f'code {program_dir}')

    # calculate the mean of precision, recall, f1 and sum of tp, tn, fp, fn
    all_dfs = pd.concat(all_dfs, ignore_index=True)
    mean_metrics = all_dfs[['precision', 'recall', 'f1', 'gt_precision', 'gt_recall', 'gt_f1']].mean().to_dict()
    sum_metrics = all_dfs[['TP', 'TN', 'FP', 'FN', 'gt_TP', 'gt_TN', 'gt_FP', 'gt_FN']].sum().to_dict()
    print(mean_metrics, sum_metrics)

    # from approach.models.hdbscan_model import HDBSCANClusterer
    # from approach.data_representation.instance_loader import load_instances

    # instances = load_instances("njr")

    # clusterer = HDBSCANClusterer(min_cluster_size=5, metric='hamming', alpha=0.5)


    # # seperate instances by thrir program
    # program_instances = defaultdict(list)
    # for inst in instances:
    #     program_instances[inst.program].append(inst)
    
    # # For each program, run the clustering
    # for program, insts in program_instances.items():
    #     print(f"Running clustering for program: {program} with {len(insts)} instances")
    #     if len(insts) < 5:
    #         print(f"Skipping program {program} due to insufficient instances ({len(insts)})")
    #         continue
    #     output_dir = f"approach/results/clustering/programwise/{program}"

    #     # Create a runner for each program
    #     runner = FlatClusteringRunner(
    #         instances=insts,
    #         clusterer=clusterer,
    #         output_dir=output_dir,
    #         run_from_main=True,
    #         only_true=True,
    #         use_trace=False,
    #         use_var=False,
    #         use_semantic=False,
    #         use_static=True,
    #     )
        
    #     # Run the clustering
    #     runner.run()