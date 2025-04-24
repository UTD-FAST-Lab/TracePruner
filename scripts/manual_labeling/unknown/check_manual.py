import os
import pandas as pd


static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'
source_code_dir = '/home/mohammad/projects/CallGraphPruner_data/njr-1_dataset/june2020_dataset'

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

    # Get library prefixes for this program
    lib_dir = os.path.join(source_code_dir, program, 'lib')
    lib_prefixes = get_lib_package_prefixes(lib_dir)

    df = pd.read_csv(csv_path)
    df_filtered = df[
        df['method'].apply(lambda m: is_application_class(m, lib_prefixes)) &
        df['target'].apply(lambda t: is_application_class(t, lib_prefixes))
    ]
    df_filtered['program'] = program
    all_edges.append(df_filtered)

# Combine all dataframes into one
combined_df = pd.concat(all_edges, ignore_index=True)

# Keep only key columns for comparison
combined_keys = combined_df[['method', 'offset', 'target']].drop_duplicates()

# === New Part: Compare with target.csv ===
target_path = '/home/mohammad/projects/CallGraphPruner/scripts/manual_labeling/unknown/edges_for_manual_labeling-42.csv'
target_df = pd.read_csv(target_path)

# Drop duplicates for fair comparison
target_keys = target_df[['method', 'offset', 'target']].drop_duplicates()

# Find rows in target that are not in combined
diff_df = target_keys.merge(combined_keys, on=['method', 'offset', 'target'], how='left', indicator=True)
missing_in_combined = diff_df[diff_df['_merge'] == 'left_only'].drop(columns=['_merge'])

# Optionally save or print
print(f"Number of rows in target.csv not in combined unknown.csv files: {len(missing_in_combined)}")
print(missing_in_combined)
# missing_in_combined.to_csv('missing_in_combined.csv', index=False)
