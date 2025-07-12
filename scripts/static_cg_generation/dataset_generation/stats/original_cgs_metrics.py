import os
import pandas as pd


auto_dcgs_path = "/20TB/mohammad/xcorpus-total-recall/dynamic_cgs"
manual_dcgs_path = "/20TB/mohammad/xcorpus-total-recall/manual_labeling2"
db_path = "/20TB/mohammad/xcorpus-total-recall/dataset"

info = {
    'doop':['v1_39', 'v3_5', 'v2_0'],
    'wala':['v1_19', 'v3_0', 'v1_23'],
    'opal':['v1_0'], 
}

programs = ['axion', 'jasml', 'xerces', 'batik']
# programs = ['axion', 'jasml', 'batik']



def evaluate_cg(dcg_df, cg_df):
    pred_edges = set(cg_df.set_index(['method', 'offset', 'target']).index)
    truth_edges = set(dcg_df.set_index(['method', 'offset', 'target']).index)
    intersection = pred_edges.intersection(truth_edges)
    precision = len(intersection) / len(pred_edges) if pred_edges else 0.0
    recall = len(intersection) / len(truth_edges) if truth_edges else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def main():

    results = []
    for tool, configs in info.items():
        for config in configs:   

            auto_metrics = []
            manual_metrics = []
            for program in programs:
                
                total_path = os.path.join(db_path, tool, 'without_jdk', program, config, 'total_edges.csv')
                unknown_path = os.path.join(db_path, tool, 'without_jdk', program, config, 'unknown_edges.csv')
                if not os.path.exists(total_path) or not os.path.exists(unknown_path):
                    continue

                total_df = pd.read_csv(total_path).drop_duplicates(subset=['method', 'offset', 'target'])
                unknown_df = pd.read_csv(unknown_path).drop_duplicates(subset=['method', 'offset', 'target'])
                if total_df.empty or unknown_df.empty:
                    continue

                # Calculate labeled edges
                labeled_df = total_df[~total_df.set_index(['method', 'offset', 'target']).index.isin(unknown_df.set_index(['method', 'offset', 'target']).index)]
                labeled_df = labeled_df.drop_duplicates(subset=['method', 'offset', 'target'])
                auto_dcg_df = pd.read_csv(os.path.join(auto_dcgs_path, program, 'dcg_filtered.csv'))
                auto_metric = evaluate_cg(auto_dcg_df, labeled_df)
                auto_metrics.append(auto_metric)
                
                # calculate the manaully labeled edges
                all_ml_edges_path = os.path.join(manual_dcgs_path, program, 'complete', 'labeling_sample.csv')
                all_ml_edges_df = pd.read_csv(all_ml_edges_path).drop_duplicates(subset=['method', 'offset', 'target'])
                # find the edges that are in both unknown and manually labeled edges
                manual_labeled_edges = all_ml_edges_df[all_ml_edges_df.set_index(['method', 'offset', 'target']).index.isin(unknown_df.set_index(['method', 'offset', 'target']).index)]
                if manual_labeled_edges.empty:
                    continue

                manual_dcg_df = pd.read_csv(os.path.join(manual_dcgs_path, program, 'dcg.csv'))
                
                # print len manual dcg_df and manual labeled edges
                print(f"Program: {program}, Manual DCG Edges: {len(manual_dcg_df)}, Manual Labeled Edges: {len(manual_labeled_edges)}")

                manual_metric = evaluate_cg(manual_dcg_df, manual_labeled_edges)
                manual_metrics.append(manual_metric)
            
            # caculate the mean of metrics
            print(auto_metrics, manual_metrics)
            auto_mean = pd.DataFrame(auto_metrics, columns=['precision', 'recall', 'f1']).mean()
            manual_mean = pd.DataFrame(manual_metrics, columns=['precision', 'recall', 'f1']).mean()

            # with 2 floating points 
            result = {
                'tool': tool,
                'config': config,
                'auto_precision': round(auto_mean['precision'], 2),
                'auto_recall': round(auto_mean['recall'], 2),
                'auto_f1': round(auto_mean['f1'], 2),
                'manual_precision': round(manual_mean['precision'], 2),
                'manual_recall': round(manual_mean['recall'], 2),
                'manual_f1': round(manual_mean['f1'], 2)
            }
            results.append(result)

    # save all results to a single csv file
    results_df = pd.DataFrame(results)
    results_df.to_csv('original_cg_stats_v2.csv', index=False)


if __name__ == "__main__":
    main()


# import os
# import pandas as pd

# # Define paths and configuration
# auto_dcgs_path = "/20TB/mohammad/xcorpus-total-recall/dynamic_cgs"
# manual_dcgs_path = "/20TB/mohammad/xcorpus-total-recall/manual_labeling2"
# db_path = "/20TB/mohammad/xcorpus-total-recall/dataset"

# info = {
#     'doop': ['v1_39', 'v3_5', 'v2_0'],
#     'wala': ['v1_19', 'v3_0', 'v1_23'],
#     'opal': ['v1_0'],
# }

# programs = ['axion', 'jasml', 'batik', 'xerces']

# def evaluate_cg(dcg_df, cg_df):
#     """Calculates precision, recall, and F1-score from ground truth and predicted edge DataFrames."""
#     # Create sets of tuples for efficient comparison
#     pred_edges = set(map(tuple, cg_df[['method', 'offset', 'target']].values))
#     truth_edges = set(map(tuple, dcg_df[['method', 'offset', 'target']].values))
    
#     intersection = pred_edges.intersection(truth_edges)
    
#     precision = len(intersection) / len(pred_edges) if pred_edges else 0.0
#     recall = len(intersection) / len(truth_edges) if truth_edges else 0.0
#     f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
#     return precision, recall, f1

# def main():
#     """
#     Main function to process tools and configurations, calculating metrics globally
#     instead of averaging per program.
#     """
#     results = []
#     for tool, configs in info.items():
#         for config in configs:
            
#             # --- Step 1: Aggregate data across all programs ---
#             all_auto_labeled_dfs = []
#             all_auto_dcg_dfs = []
#             all_manual_labeled_dfs = []
#             all_manual_dcg_dfs = []

#             for program in programs:
#                 total_path = os.path.join(db_path, tool, 'without_jdk', program, config, 'total_edges.csv')
#                 unknown_path = os.path.join(db_path, tool, 'without_jdk', program, config, 'unknown_edges.csv')

#                 if not os.path.exists(total_path) or not os.path.exists(unknown_path):
#                     continue

#                 total_df = pd.read_csv(total_path)
#                 unknown_df = pd.read_csv(unknown_path)
                
#                 # --- Auto-labeled edges ---
#                 labeled_df = total_df[~total_df.set_index(['method', 'offset', 'target']).index.isin(unknown_df.set_index(['method', 'offset', 'target']).index)]
#                 if not labeled_df.empty:
#                     auto_dcg_df = pd.read_csv(os.path.join(auto_dcgs_path, program, 'dcg_filtered.csv'))
#                     all_auto_labeled_dfs.append(labeled_df)
#                     all_auto_dcg_dfs.append(auto_dcg_df)

#                 # --- Manually-labeled edges ---
#                 all_ml_edges_path = os.path.join(manual_dcgs_path, program, 'labeling_sample.csv')
#                 if os.path.exists(all_ml_edges_path):
#                     all_ml_edges_df = pd.read_csv(all_ml_edges_path)
#                     manual_labeled_edges = all_ml_edges_df[all_ml_edges_df.set_index(['method', 'offset', 'target']).index.isin(unknown_df.set_index(['method', 'offset', 'target']).index)]
#                     if not manual_labeled_edges.empty:
#                         manual_dcg_df = pd.read_csv(os.path.join(manual_dcgs_path, program, 'dcg.csv'))
#                         all_manual_labeled_dfs.append(manual_labeled_edges)
#                         all_manual_dcg_dfs.append(manual_dcg_df)

#             # --- Step 2: Calculate metrics on aggregated data ---
#             auto_precision, auto_recall, auto_f1 = (0.0, 0.0, 0.0)
#             if all_auto_labeled_dfs:
#                 # Combine all dataframes and remove duplicates across the entire dataset
#                 aggregated_auto_labeled = pd.concat(all_auto_labeled_dfs).drop_duplicates(subset=['method', 'offset', 'target'])
#                 aggregated_auto_dcg = pd.concat(all_auto_dcg_dfs).drop_duplicates(subset=['method', 'offset', 'target'])
#                 auto_precision, auto_recall, auto_f1 = evaluate_cg(aggregated_auto_dcg, aggregated_auto_labeled)
            
#             manual_precision, manual_recall, manual_f1 = (0.0, 0.0, 0.0)
#             if all_manual_labeled_dfs:
#                 # Combine all dataframes and remove duplicates
#                 aggregated_manual_labeled = pd.concat(all_manual_labeled_dfs).drop_duplicates(subset=['method', 'offset', 'target'])
#                 aggregated_manual_dcg = pd.concat(all_manual_dcg_dfs).drop_duplicates(subset=['method', 'offset', 'target'])
#                 manual_precision, manual_recall, manual_f1 = evaluate_cg(aggregated_manual_dcg, aggregated_manual_labeled)

#             # --- Step 3: Store results ---
#             result = {
#                 'tool': tool, 'config': config,
#                 'auto_precision': round(auto_precision, 2),
#                 'auto_recall': round(auto_recall, 2),
#                 'auto_f1': round(auto_f1, 2),
#                 'manual_precision': round(manual_precision, 2),
#                 'manual_recall': round(manual_recall, 2),
#                 'manual_f1': round(manual_f1, 2)
#             }
#             results.append(result)

#     # Save all results to a single CSV file
#     results_df = pd.DataFrame(results)
#     results_df.to_csv('global_3cg_stats.csv', index=False)
#     print("Saved global results to global_3cg_stats.csv")

# if __name__ == "__main__":
#     main()