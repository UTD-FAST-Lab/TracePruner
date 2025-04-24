# import os
# import pandas as pd

# # Configurations
# output_csv = "edges_for_manual_labeling-40.csv"
# sample_size = 50
# static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'
# source_code_dir = '/home/mohammad/projects/CallGraphPruner_data/njr-1_dataset/june2020_dataset'

# # Get top-level library package names from the libs directory
# def get_lib_package_prefixes(lib_dir):
#     if not os.path.isdir(lib_dir):
#         return set()
#     return set(os.listdir(lib_dir))  # e.g., {'org', 'com', 'javax'}

# # Check if name starts with any library prefix
# def is_application_class(name, lib_prefixes):
#     return all(not name.startswith(prefix + '/') for prefix in lib_prefixes)

# # Collect all filtered unknown edges
# all_edges = []

# for program in os.listdir(static_cg_dir):
#     csv_path = os.path.join(static_cg_dir, program, 'unknown.csv')
#     if not os.path.exists(csv_path):
#         continue

#     # Get library prefixes for this program
#     lib_dir = os.path.join(source_code_dir, program, 'lib')
#     lib_prefixes = get_lib_package_prefixes(lib_dir)

#     df = pd.read_csv(csv_path)
#     df_filtered = df[
#         df['method'].apply(lambda m: is_application_class(m, lib_prefixes)) &
#         df['target'].apply(lambda t: is_application_class(t, lib_prefixes))
#     ]
#     df_filtered['program'] = program
#     all_edges.append(df_filtered)

# # Combine all dataframes
# combined_df = pd.concat(all_edges, ignore_index=True)

# # Sample 50 edges
# df_sampled = combined_df.sample(n=sample_size, random_state=41)
# # Keep only key columns for comparison
# df_sampled = df_sampled[['program', 'method', 'offset', 'target']]

# # Add empty label column
# df_sampled['label'] = ""

# # Reorder columns
# cols = ['program'] + [col for col in df_sampled.columns if col not in ['program', 'label']] + ['label']
# df_sampled = df_sampled[cols]

# # Sort by program
# df_sampled = df_sampled.sort_values(by='program')

# # Save to CSV
# df_sampled.to_csv(output_csv, index=False)
# print(f"✅ Sampled {sample_size} filtered application edges saved to {output_csv}")


import os
import pandas as pd

# Configurations
static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'
source_code_dir = '/home/mohammad/projects/CallGraphPruner_data/njr-1_dataset/june2020_dataset'
batches_dir = './batches'  # folder containing previous batch CSVs
batch_output_prefix = "edges_batch"  # new batch file prefix
num_samples_total = 400
batch_size = 50

# Get top-level library package names from the libs directory
def get_lib_package_prefixes(lib_dir):
    if not os.path.isdir(lib_dir):
        return set()
    return set(os.listdir(lib_dir))  # e.g., {'org', 'com', 'javax'}

# Check if name starts with any library prefix
def is_application_class(name, lib_prefixes):
    return all(not name.startswith(prefix + '/') for prefix in lib_prefixes)

# Collect all filtered unknown edges
all_edges = []

for program in os.listdir(static_cg_dir):
    csv_path = os.path.join(static_cg_dir, program, 'unknown.csv')
    if not os.path.exists(csv_path):
        continue

    lib_dir = os.path.join(source_code_dir, program, 'lib')
    lib_prefixes = get_lib_package_prefixes(lib_dir)

    df = pd.read_csv(csv_path)
    df_filtered = df[
        df['method'].apply(lambda m: is_application_class(m, lib_prefixes)) &
        df['target'].apply(lambda t: is_application_class(t, lib_prefixes))
    ]
    df_filtered['program'] = program
    all_edges.append(df_filtered)

combined_df = pd.concat(all_edges, ignore_index=True)
combined_df = combined_df[['program', 'method', 'offset', 'target']]

# --- Filter out previously selected edges ---
existing_edges = set()

if os.path.isdir(batches_dir):
    for fname in os.listdir(batches_dir):
        if not fname.endswith('.csv'):
            continue
        batch_df = pd.read_csv(os.path.join(batches_dir, fname))
        for _, row in batch_df.iterrows():
            key = (row['program'], row['method'], row['offset'], row['target'])
            existing_edges.add(key)

# Remove duplicates
combined_df['edge_key'] = combined_df.apply(lambda row: (row['program'], row['method'], row['offset'], row['target']), axis=1)
combined_df = combined_df[~combined_df['edge_key'].isin(existing_edges)]

# Shuffle and take N samples
sampled_df = combined_df.sample(n=num_samples_total, random_state=42).reset_index(drop=True)
sampled_df = sampled_df.drop(columns=['edge_key'])

# Add label column
sampled_df['label'] = ""

# Reorder columns
cols = ['program', 'method', 'offset', 'target', 'label']
sampled_df = sampled_df[cols]

# Create 50-sample batches
os.makedirs("new_batches", exist_ok=True)
num_batches = num_samples_total // batch_size

for i in range(num_batches):
    start = i * batch_size
    end = start + batch_size
    batch = sampled_df.iloc[start:end]
    batch_name = f"new_batches/{batch_output_prefix}{i+3}.csv"  # Assuming batch1 and batch2 already exist
    batch.to_csv(batch_name, index=False)
    print(f"✅ Saved batch {i+3} with {len(batch)} samples to {batch_name}")
