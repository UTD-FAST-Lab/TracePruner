import os
import pandas as pd

# Load all unknown edges
all_unknown = []
unknown_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/static_cgs_no_exlusion'

for program in os.listdir(unknown_dir):
    csv_path = os.path.join(unknown_dir, program, 'unknown.csv')
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    df['program'] = program  # Add program name as a column
    all_unknown.append(df)

combined_df = pd.concat(all_unknown, ignore_index=True)
combined_df = combined_df[['program', 'method', 'offset', 'target']]
combined_df.drop_duplicates(inplace=True)

# Load all batch edges
all_batches = []
batch_dir = '/home/mohammad/projects/CallGraphPruner/scripts/manual_labeling/unknown/new_batches'

for file in os.listdir(batch_dir):
    if not file.endswith('.csv'):
        continue

    df = pd.read_csv(os.path.join(batch_dir, file))
    df = df[['method', 'offset', 'target']]
    all_batches.append(df)

combined_batches = pd.concat(all_batches, ignore_index=True)
combined_batches.drop_duplicates(inplace=True)

# Find edges in batches that are NOT in the unknowns
unmatched = pd.merge(combined_batches, combined_df[['method', 'offset', 'target']], 
                     on=['method', 'offset', 'target'], 
                     how='left', 
                     indicator=True)

unmatched = unmatched[unmatched['_merge'] == 'left_only']
unmatched.drop(columns=['_merge'], inplace=True)

print(f"✅ Saved {len(unmatched)} unmatched edges to unmatched_batches.csv")
