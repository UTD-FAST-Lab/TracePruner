import pandas as pd
from pathlib import Path

# Paths
program_list_file = '/home/mohammad/projects/CallGraphPruner/data/programs/all_programs.txt'  # Path to the list of program names
data_directory = Path('/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9')  # Replace this with the actual path to your programs
cgpruner_directory = Path('/home/mohammad/projects/CallGraphPruner_data/dataset-high-precision-callgraphs/full_callgraphs_set')  # Path to the CGPruner data
# output_directory = Path('output_differences')
# output_directory.mkdir(exist_ok=True)

# Read list of programs
with open(program_list_file, 'r') as file:
    program_names = [line.strip() for line in file if line.strip()]

for program in program_names:
    data_program_dir = data_directory / program
    cgpruner_program_dir = cgpruner_directory / program

    cgpruner_path = cgpruner_program_dir / 'wala0cfa.csv'
    trace_path = data_program_dir / 'wala0cfa.csv'

    output_path = data_program_dir / 'wala0cfa_filtered.csv'



    # Load both CSVs
    cgpruner_df = pd.read_csv(cgpruner_path)
    trace_df = pd.read_csv(trace_path)

    # Reduce both to key columns for matching
    cgpruner_keys = cgpruner_df[['method', 'offset', 'target']]
    trace_keys = trace_df[['method', 'offset', 'target']]

    # Perform an inner merge to find the intersection
    intersection = pd.merge(
        cgpruner_df[['method', 'offset', 'target', 'wiretap']],
        trace_keys,
        on=['method', 'offset', 'target'],
        how='inner'
    )

    # Save to filtered.csv
    intersection.to_csv(output_path, index=False)
