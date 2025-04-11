import pandas as pd
from pathlib import Path

# Paths
program_list_file = '/home/mohammad/projects/CallGraphPruner/data/programs/all_programs.txt'  # Path to the list of program names
base_directory = Path('/home/mohammad/projects/CallGraphPruner_data/dataset-high-precision-callgraphs/full_callgraphs_set')  # Replace this with the actual path to your programs
output_directory = Path('/home/mohammad/projects/CallGraphPruner/data/static-cgs')

# Read list of programs
with open(program_list_file, 'r') as file:
    program_names = [line.strip() for line in file if line.strip()]

for program in program_names:
    program_dir = base_directory / program
    file_0cfa = program_dir / 'wala0cfa.csv'

    if not file_0cfa.exists():
        print(f"Skipping {program}: missing 0cfa.csv")
        continue

    # Load and filter rows where method or target start with "java/"
    df_0 = pd.read_csv(file_0cfa)

    df_0 = df_0[~df_0['method'].str.startswith('java/') & ~df_0['target'].str.startswith('java/') & df_0['wiretap']==1]

    # Drop duplicates based on (method, offset, target)
    df_0 = df_0.drop_duplicates(subset=['method', 'offset', 'target'])

    df_0 = df_0[['method', 'offset', 'target']]

    output_file = output_directory / program
    df_0.to_csv(output_file / f"true_edges.csv", index=False)