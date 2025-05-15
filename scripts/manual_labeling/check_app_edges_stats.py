
import os
import pandas as pd

# Configurations
# static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'
static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/static_cgs_no_exlusion'

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
    # csv_path = os.path.join(static_cg_dir, program, 'diff_0cfa_1obj.csv')
    # csv_path = os.path.join(static_cg_dir, program, 'true_edges.csv')
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
combined_df.drop_duplicates(inplace=True)

print("Total number of edges:", len(combined_df))