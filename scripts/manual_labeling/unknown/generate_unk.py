import os
import pandas as pd




programs_path = '/home/mohammad/projects/CallGraphPruner/data/programs/all_programs.txt' 

static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'
static_cg_dir_2 = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/static_cgs_no_exlusion'

with open(programs_path, 'r') as f:
    program_names = [line.strip() for line in f if line.strip()]


for program in program_names:
    program_dir_path = os.path.join(static_cg_dir, program)
    program_dir_path_2 = os.path.join(static_cg_dir_2, program)
    true_path = os.path.join(program_dir_path, 'true_edges.csv')
    false_path = os.path.join(program_dir_path_2, 'diff_0cfa_1cfa.csv')
    all_edges_path = os.path.join(program_dir_path, 'wala0cfa_filtered.csv')
    
    static_features_path = os.path.join(program_dir_path, 'static_featuers', 'wala0cfa_filtered.csv')


    true_df = pd.read_csv(true_path)
    false_df = pd.read_csv(false_path) if os.path.exists(false_path) else pd.DataFrame(columns=['method', 'offset', 'target'])
    all_df = pd.read_csv(all_edges_path)


    all_df = all_df[~all_df['method'].str.startswith('java/') & ~all_df['target'].str.startswith('java/')]

    labeled_df = pd.concat([true_df, false_df])
    labeled_keys = labeled_df[['method', 'offset', 'target']].drop_duplicates()
    unknown_df = all_df.merge(labeled_keys, on=['method', 'offset', 'target'], how='left', indicator=True)
    unknown_df = unknown_df[unknown_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    unknown_df.to_csv(f'{program_dir_path_2}/unknown.csv', index=False)