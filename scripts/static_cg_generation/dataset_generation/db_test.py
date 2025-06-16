import os
import pandas as pd


programs = [
    'axion',
    'batik',
    'xerces',
    'jasml'
]

config_id = 'v1_39'
tool = 'doop'


scg_path = '/20TB/mohammad/xcorpus-total-recall/dataset'
output = '/20TB/mohammad/xcorpus-total-recall/test'

for program in programs:
    program_path = os.path.join(scg_path, tool, 'without_jdk', program, config_id)

    total_df = pd.read_csv(os.path.join(program_path, 'total_edges.csv'))
    true_df = pd.read_csv(os.path.join(program_path, 'true_edges.csv'))
    false_df = pd.read_csv(os.path.join(program_path, 'false_edges.csv'))
    # unknown_df = pd.read_csv(os.path.join(program_path, 'unknown_edges.csv'))

    # add a column to totaldf called label.
    key = ['method', 'offset', 'target']
    total_df['label'] = -1
    total_df.loc[total_df[key].apply(tuple, 1).isin(true_df[key].apply(tuple, 1)), 'label'] = 1
    total_df.loc[total_df[key].apply(tuple, 1).isin(false_df[key].apply(tuple, 1)), 'label'] = 0

    # save to csv
    if not os.path.exists(os.path.join(output, program)):
        os.makedirs(os.path.join(output, program))
    total_df.to_csv(os.path.join(output, program, 'wala0cfa.csv'), index=False)

    print(f"Processed {program} and saved to {os.path.join(output, program, 'wala0cfa.csv')}")


