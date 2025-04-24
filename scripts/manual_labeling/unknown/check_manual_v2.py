import os
import pandas as pd


all_manual_labeling = pd.read_csv('/home/mohammad/projects/CallGraphPruner/data/manual/manual_unknown_labels.csv')
new_file = pd.read_csv('edges_for_manual_labeling-40.csv')

# Keep only key columns for comparison
combined_keys = all_manual_labeling[['method', 'offset', 'target']].drop_duplicates()

# Drop duplicates for fair comparison
target_keys = new_file[['method', 'offset', 'target']].drop_duplicates()

# Find rows in target that are not in combined
diff_df = target_keys.merge(combined_keys, on=['method', 'offset', 'target'], how='left', indicator=True)
missing_in_combined = diff_df[diff_df['_merge'] == 'left_only'].drop(columns=['_merge'])

# Optionally save or print
print(f"Number of rows in target.csv not in combined unknown.csv files: {len(missing_in_combined)}")
print(missing_in_combined)
# missing_in_combined.to_csv('missing_in_combined.csv', index=False)
