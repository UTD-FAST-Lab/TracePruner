
import os
import pandas as pd

# Configurations
static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'
static_cg_dir_false = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/static_cgs_no_exlusion'
source_code_dir = '/home/mohammad/projects/CallGraphPruner_data/njr-1_dataset/june2020_dataset'

# Get top-level library package names from the libs directory
def get_lib_package_prefixes(lib_dir):
    if not os.path.isdir(lib_dir):
        return set()

    prefixes = set()
    for root, dirs, files in os.walk(lib_dir):
        for file in files:
            if file.endswith(".class"):
                # Get the relative path from the lib directory
                relative_path = os.path.relpath(os.path.join(root, file), lib_dir)
                
                # Replace file separator with / and remove the .class extension
                package_name = relative_path.replace(os.path.sep, "/").replace(".class", "")
                
                # Add the full package prefix
                prefixes.add(package_name)

    return prefixes

# Check if name starts with any library prefix
def is_application_class(name, lib_prefixes):
    return all(not name.startswith(prefix) for prefix in lib_prefixes)


for program in os.listdir(static_cg_dir):
    # csv_path = os.path.join(static_cg_dir, program, 'wala0cfa_filtered.csv')
    # csv_path = os.path.join(static_cg_dir, program, 'true_edges.csv')
    csv_path = os.path.join(static_cg_dir_false, program, 'diff_0cfa_1obj.csv')

    if not os.path.exists(csv_path):
        continue

    lib_dir = os.path.join(source_code_dir, program, 'lib')
    lib_prefixes = get_lib_package_prefixes(lib_dir)

    # print(lib_prefixes)

    df = pd.read_csv(csv_path)
    df_filtered = df[
        df['method'].apply(lambda m: is_application_class(m, lib_prefixes)) &
        df['target'].apply(lambda t: is_application_class(t, lib_prefixes))
    ]

    # output_path = os.path.join(static_cg_dir, program, 'wala0cfa_filtered_libs.csv')
    # output_path = os.path.join(static_cg_dir, program, 'true_filtered_libs.csv')
    output_path = os.path.join(static_cg_dir_false, program, 'false_filtered_libs.csv')
    df_filtered.to_csv(output_path, index=False)