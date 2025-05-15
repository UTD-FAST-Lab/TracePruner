import os
import pandas as pd
from pathlib import Path

# Paths
program_list_file = '/home/mohammad/projects/CallGraphPruner/data/programs/all_programs.txt'
base_directory = Path('/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9')
no_ex_base_directory = Path('/home/mohammad/projects/CallGraphPruner/data/static-cgs/static_cgs_no_exlusion')

# Read list of programs
with open(program_list_file, 'r') as file:
    program_names = [line.strip() for line in file if line.strip()]

for program in program_names:
    program_dir_0cfa = base_directory / program
    program_dir_1cfa = no_ex_base_directory / program
    file_0cfa = program_dir_0cfa / 'wala0cfa_filtered.csv'
    file_1cfa = program_dir_1cfa / 'wala1obj.csv'

    if not file_0cfa.exists() or not file_1cfa.exists():
        print(f"Skipping {program}: missing 0cfa or 1cfa file")
        continue

    # Load and filter rows where method or target start with "java/"
    df_0 = pd.read_csv(file_0cfa)
    df_1 = pd.read_csv(file_1cfa)

    df_0 = df_0[~df_0['method'].str.startswith('java/') & ~df_0['target'].str.startswith('java/')]
    df_1 = df_1[~df_1['method'].str.startswith('java/') & ~df_1['target'].str.startswith('java/')]

    # Drop duplicates based on (method, offset, target)
    df_0 = df_0.drop_duplicates(subset=['method', 'offset', 'target'])
    df_1 = df_1.drop_duplicates(subset=['method', 'offset', 'target'])

    # Find rows in df_0 that are not in df_1
    diff = pd.merge(df_0, df_1, on=['method', 'offset', 'target'], how='left', indicator=True)
    diff = diff[diff['_merge'] == 'left_only']
    diff = diff[['method', 'offset', 'target']]

    # Save to file if there are any differences
    if not diff.empty:
        diff_file_path = program_dir_1cfa / "diff_0cfa_1obj.csv"
        diff.to_csv(diff_file_path, index=False)
        print(f"✅ Differences for {program} saved to {diff_file_path}")
