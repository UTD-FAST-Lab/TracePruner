from approach.Instance import Instance
import pandas as pd
import os


import os
import pandas as pd

def load_semantic_features(semantic_features_dir):
    test_df = pd.read_csv(os.path.join(semantic_features_dir, 'enriched_ft_test.csv'))
    train_df = pd.read_csv(os.path.join(semantic_features_dir, 'enriched_ft_train.csv'))

    combined_df = pd.concat([test_df, train_df], ignore_index=True)

    semantic_map = {}

    for _, row in combined_df.iterrows():
        key = (
            row['program_name'],
            row['method'],
            row['offset'],
            row['target']
        )
        try:
            code_str = row['code']
            code_vec = [float(x) for x in code_str.strip().split(',')]
            semantic_map[key] = code_vec
        except Exception:
            print(f"Error processing row: {row}")

    return semantic_map



def load_instances(dataset="njr"):
    assert dataset == "njr", "Only njr dataset is currently supported."
    
    programs_path = '/home/mohammad/projects/CallGraphPruner/data/programs/all_programs.txt' 
    static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'
    manual_gt_path = '/home/mohammad/projects/CallGraphPruner/data/manual/manual_unknown_labels.csv'

    semantic_features_dir = '/home/mohammad/projects/CallGraphPruner_data/autoPruner/'
    
    with open(programs_path, 'r') as f:
        program_names = [line.strip() for line in f if line.strip()]

    # Load manually labeled unknowns
    manual_gt_map = {}
    if os.path.exists(manual_gt_path):
        manual_gt_df = pd.read_csv(manual_gt_path)
        for _, row in manual_gt_df.iterrows():
            key = (row['program'], row['method'], row['offset'], row['target'])
            manual_gt_map[key] = int(row['label'])  # 0 or 1

    
    # load and create the df for semantic features
    semantic_feature_map = load_semantic_features(semantic_features_dir)


    instances = []
    for program in program_names:
        program_dir_path = os.path.join(static_cg_dir, program)
        true_path = os.path.join(program_dir_path, 'true_edges.csv')
        false_path = os.path.join(program_dir_path, 'diff_0cfa_1cfa.csv')
        all_edges_path = os.path.join(program_dir_path, 'wala0cfa_filtered.csv')
        static_features_path = os.path.join(program_dir_path, 'static_featuers', 'wala0cfa_filtered.csv')

        if not os.path.exists(true_path) or not os.path.exists(all_edges_path) or not os.path.exists(static_features_path):
            continue

        true_df = pd.read_csv(true_path)
        false_df = pd.read_csv(false_path) if os.path.exists(false_path) else pd.DataFrame(columns=['method', 'offset', 'target'])
        all_df = pd.read_csv(all_edges_path)
        static_features_df = pd.read_csv(static_features_path).set_index(['method', 'offset', 'target'])

        all_df = all_df[~all_df['method'].str.startswith('java/') & ~all_df['target'].str.startswith('java/')]

        labeled_df = pd.concat([true_df, false_df])
        labeled_keys = labeled_df[['method', 'offset', 'target']].drop_duplicates()
        unknown_df = all_df.merge(labeled_keys, on=['method', 'offset', 'target'], how='left', indicator=True)
        unknown_df = unknown_df[unknown_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        # unknown_df.to_csv(f'{program_dir_path}/unknown.csv', index=False)

        def create_instance(row, label, is_unknown):
            key = (row['method'], row['offset'], row['target'])
            if key in static_features_df.index:
                inst = Instance(program, *key, is_unknown, label=label)
                inst.set_static_features(static_features_df.loc[key])

                # Set semantic features if available
                semantic_key = (program, row['method'], row['offset'], row['target'])
                if semantic_key in semantic_feature_map:
                    inst.set_semantic_features(semantic_feature_map[semantic_key])
                else:
                    print(f"Semantic features not found for {semantic_key}")

                 # Set GT if exists
                gt_key = (program, row['method'], row['offset'], row['target'])
                if gt_key in manual_gt_map:
                    inst.set_ground_truth(manual_gt_map[gt_key])
                
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
