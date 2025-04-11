import pandas as pd
from pathlib import Path

# Paths
program_list_file = '/home/mohammad/projects/CallGraphPruner/data/programs/all_programs.txt'  # Path to the list of program names
base_directory = Path('/home/mohammad/projects/CallGraphPruner/data/static-cgs')  # Replace this with the actual path to your programs
# output_directory = Path('output_differences')
# output_directory.mkdir(exist_ok=True)

# Read list of programs
with open(program_list_file, 'r') as file:
    program_names = [line.strip() for line in file if line.strip()]

for program in program_names:
    program_dir = base_directory / program
    file_0cfa = program_dir / 'wala0cfa.csv'
    # file_1cfa = program_dir / 'wala1cfa.csv'
    file_1cfa = program_dir / 'wala1obj.csv'

    if not file_0cfa.exists() or not file_1cfa.exists():
        print(f"Skipping {program}: missing 0cfa.csv or 1cfa.csv")
        continue

    # Load and filter rows where method or target start with "java/"
    df_0 = pd.read_csv(file_0cfa)
    df_1 = pd.read_csv(file_1cfa)

    df_0 = df_0[~df_0['method'].str.startswith('java/') & ~df_0['target'].str.startswith('java/')]
    df_1 = df_1[~df_1['method'].str.startswith('java/') & ~df_1['target'].str.startswith('java/')]

    # Drop duplicates based on (method, offset, target)
    df_0 = df_0.drop_duplicates(subset=['method', 'offset', 'target'])
    df_1 = df_1.drop_duplicates(subset=['method', 'offset', 'target'])

    # Add marker columns to each
    df_0['0cfa'] = 1
    df_0['1obj'] = 0

    df_1['0cfa'] = 0
    df_1['1obj'] = 1

    # Merge on method, offset, target to find symmetric difference
    merged = pd.merge(df_0, df_1, on=['method', 'offset', 'target'], how='outer', indicator=True)

    # Keep only the rows that are in one file but not both
    diff = merged[merged['_merge'] != 'both']

    # Clean up
    diff = diff[['method', 'offset', 'target', '0cfa_x', '1obj_x', '0cfa_y', '1obj_y']]
    diff['0cfa'] = diff['0cfa_x'].fillna(diff['0cfa_y']).astype(int)
    diff['1obj'] = diff['1obj_x'].fillna(diff['1obj_y']).astype(int)
    diff = diff[['method', 'offset', 'target', '0cfa', '1obj']]

    # Save to file if there are any differences
    if not diff.empty:
        diff.to_csv(program_dir / f"diff_0cfa_1obj.csv", index=False)
