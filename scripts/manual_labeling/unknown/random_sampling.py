import os
import pandas as pd

# Configurations
output_csv = "edges_for_manual_labeling-41.csv"
sample_size = 50
static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs'
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

# Combine all dataframes
combined_df = pd.concat(all_edges, ignore_index=True)

# Sample 50 edges
df_sampled = combined_df.sample(n=sample_size, random_state=41)

# Add empty label column
df_sampled['label'] = ""

# Reorder columns
cols = ['program'] + [col for col in df_sampled.columns if col not in ['program', 'label']] + ['label']
df_sampled = df_sampled[cols]

# Sort by program
df_sampled = df_sampled.sort_values(by='program')

# Save to CSV
df_sampled.to_csv(output_csv, index=False)
print(f"✅ Sampled {sample_size} filtered application edges saved to {output_csv}")
