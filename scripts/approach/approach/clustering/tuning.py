
from approach.clustering.flat_clustering_runner import FlatClusteringRunner
from approach.models.hdbscan_model import HDBSCANClusterer
from approach.data_representation.instance_loader import load_instances

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns



def main():
   # Load instances
    instances = load_instances("njr")
    all_runners = []


    min_cluster_sizes = [5, 10, 20, 50]
    min_samples_list = [1, 2, 5, 10, 20]
    cluster_selection_methods = ['eom', 'leaf']
    metrics = ['euclidean', 'manhattan', 'hamming']
    alpha = [0.5, 1.0, 1.5, 2.0]

    # create the product of all parameters

    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_list:
            for cluster_selection_method in cluster_selection_methods:
                for alpha_value in alpha:
                    for metric in metrics:

                        option = f"{min_cluster_size}_{min_samples}_{cluster_selection_method}_{alpha_value}_{metric}"

                        # create a new instance of the HDBSCANClusterer with the current parameters
                        clusterer = HDBSCANClusterer(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_method=cluster_selection_method,
                            metric=metric,
                            alpha=alpha_value
                        )
                        # create a new instance of the FlatClusteringRunner with the current clusterer
                        runner = FlatClusteringRunner(
                            instances=instances,
                            clusterer=clusterer,
                            only_true=True,
                            output_dir="approach/results/clustering",
                            use_trace=True,
                            use_semantic=False,
                            use_static=False,
                            run_from_main=True,
                            option=option,
                        )
                        all_runners.append(runner)


    for runner in all_runners:
        # run the clustering
        runner.run()




def load_all_results(result_dir):
    all_metrics = []
    for file in os.listdir(result_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(result_dir, file))
            df["config_file"] = file  # Track which file this came from
            all_metrics.append(df)
    
    # Combine into a single DataFrame
    combined_df = pd.concat(all_metrics, ignore_index=True)
    return combined_df

def balanced_score(row):
    # Balance regular and GT scores
    regular_f1 = row["f1"]
    gt_f1 = row["gt_f1"]
    regular_tp = row["TP"]
    gt_tp = row["gt_TP"]
    regular_tn = row["TN"]
    gt_tn = row["gt_TN"]

    # We want both regular and GT F1 to be high, and both TPs to be high
    return 0.2 * regular_f1 + 0.6 * gt_f1 + 0 * regular_tp + 1 * gt_tp + 0.2 * regular_tn + 4 * gt_tn


def plot_all_configs(results_df, output_file="all_configs_plot.png"):
    # Plot
    plt.figure(figsize=(18, 10))
    # sns.barplot(
    #     x="config_file", 
    #     y="TP", 
    #     data=results_df, 
    #     color="green", 
    #     label="Regular TP"
    # )
    # sns.barplot(
    #     x="config_file", 
    #     y="TN", 
    #     data=results_df, 
    #     color="lightgreen", 
    #     label="Regular TN"
    # )
    sns.barplot(
        x="config_file", 
        y="gt_TP", 
        data=results_df, 
        color="blue", 
        label="GT TP"
    )
    sns.barplot(
        x="config_file", 
        y="gt_TN", 
        data=results_df, 
        color="lightblue", 
        label="GT TN"
    )

    plt.xticks(rotation=90)
    plt.title("Regular vs GT TP/TN for All Configs")
    plt.ylabel("Count")
    plt.xlabel("Config File")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def find_the_best_config():

    result_dir = "approach/results/clustering/only_true/tuning"
    results_df  = load_all_results(result_dir)

    # filter out the configs that have lower than .7 f1 score for gt
    results_df = results_df[results_df["gt_f1"] > 0.7]


    # Plot all configs
    plot_all_configs(results_df, output_file="approach/results/clustering/only_true/tuning/all_configs_plot.png")

    # results_df["balanced_score"] = results_df.apply(balanced_score, axis=1)
    # top_10 = results_df.sort_values("balanced_score", ascending=False).head(20)

    # print(top_10[["config_file", "precision", "recall", "f1", "gt_precision", "gt_recall", "gt_f1", "TP", "gt_TP"]])
    # # Save the top 10 results to a CSV file
    # top_10.to_csv("approach/results/clustering/only_true/struct_top_10_balanced_configs.csv", index=False)


if __name__ == "__main__":
    main()
    # find_the_best_config()