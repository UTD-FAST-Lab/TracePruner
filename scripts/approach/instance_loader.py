from approach.Instance import Instance
import pandas as pd
import os

def load_instances(dataset="njr"):
    assert dataset == "njr", "Only njr dataset is currently supported."
    
    programs_path = '/home/mohammad/projects/CallGraphPruner/data/programs/all_programs.txt' 
    static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs'
    
    with open(programs_path, 'r') as f:
        program_names = [line.strip() for line in f if line.strip()]

    instances = []
    for program in program_names:
        program_dir_path = os.path.join(static_cg_dir, program)
        true_path = os.path.join(program_dir_path, 'true_edges.csv')
        false_path = os.path.join(program_dir_path, 'diff_0cfa_1obj.csv')
        all_edges_path = os.path.join(program_dir_path, 'wala0cfa.csv')
        features_path = os.path.join(program_dir_path, 'static_featuers', 'wala0cfa.csv')

        if not os.path.exists(true_path) or not os.path.exists(all_edges_path) or not os.path.exists(features_path):
            continue

        true_df = pd.read_csv(true_path)
        false_df = pd.read_csv(false_path) if os.path.exists(false_path) else pd.DataFrame(columns=['method', 'offset', 'target'])
        all_df = pd.read_csv(all_edges_path)
        features_df = pd.read_csv(features_path).set_index(['method', 'offset', 'target'])

        all_df = all_df[~all_df['method'].str.startswith('java/') & ~all_df['target'].str.startswith('java/')]

        labeled_df = pd.concat([true_df, false_df])
        labeled_keys = labeled_df[['method', 'offset', 'target']].drop_duplicates()
        unknown_df = all_df.merge(labeled_keys, on=['method', 'offset', 'target'], how='left', indicator=True)
        unknown_df = unknown_df[unknown_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        unknown_df.to_csv(f'{program_dir_path}/unknown.csv', index=False)

        def create_instance(row, label, is_unknown):
            key = (row['method'], row['offset'], row['target'])
            if key in features_df.index:
                inst = Instance(program, *key, is_unknown, label=label)
                inst.set_static_features(features_df.loc[key])
                return inst

        for _, row in true_df.iterrows():
            inst = create_instance(row, label=True, is_unknown=False)
            if inst: instances.append(inst)

        for _, row in false_df.iterrows():
            inst = create_instance(row, label=False, is_unknown=False)
            if inst: instances.append(inst)

        for _, row in unknown_df.iterrows():
            inst = create_instance(row, label=False, is_unknown=True)
            if inst: instances.append(inst)

    return instances
