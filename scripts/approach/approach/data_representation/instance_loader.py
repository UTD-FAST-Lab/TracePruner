from approach.data_representation.Instance import Instance
from approach.data_representation.utils import load_finetuned_semantic_features
import pandas as pd
import os


# CONFIGURATION
LOAD_TRACE = False
LOAD_VAR = False

LOAD_SEMANTIC_INCORRECT_FINETUNED = False
LOAD_SEMANTIC_FINETUNED = False
LOAD_SEMANTIC_RAW = False

IGNORE_LIBS = True


def load_instances_njr(dataset="njr"):
    assert dataset == "njr", "Only njr dataset is currently supported."
    
    programs_path = '/home/mohammad/projects/CallGraphPruner/data/programs/all_programs.txt' 
    manual_gt_path = '/home/mohammad/projects/CallGraphPruner/data/manual/manual_unknown_labels.csv'
    
    static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'
    static_cg_dir_false = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/static_cgs_no_exlusion'

    if LOAD_SEMANTIC_RAW:
        semantic_features_dir = '/home/mohammad/projects/CallGraphPruner/data/semantic_embeddings/semantic_embeddings'

    elif LOAD_SEMANTIC_INCORRECT_FINETUNED:
        semantic_features_dir = '/home/mohammad/projects/CallGraphPruner_data/autoPruner'
        semantic_feature_map = load_finetuned_semantic_features(semantic_features_dir)

    elif LOAD_SEMANTIC_FINETUNED:
        semantic_features_dir = '/home/mohammad/projects/CallGraphPruner/data/semantic_embeddings/semantic_finetuned'

    var_features_dir = '/home/mohammad/projects/CallGraphPruner/data/trace-embeddings/var_embeddings/var_embeddings'   # varinfo
    trace_features_dir = '/home/mohammad/projects/CallGraphPruner/data/trace-embeddings/n2v/sum/tfidf'   # n2v
    # trace_features_dir = '/home/mohammad/projects/CallGraphPruner/data/trace-embeddings/gnn/cg_embeddings_dgi_weighted'   # gnn
    
    with open(programs_path, 'r') as f:
        program_names = [line.strip() for line in f if line.strip()]

    # Load manually labeled unknowns
    manual_gt_map = {}
    if os.path.exists(manual_gt_path):
        manual_gt_df = pd.read_csv(manual_gt_path)
        for _, row in manual_gt_df.iterrows():
            key = (row['program'], row['method'], row['offset'], row['target'])
            manual_gt_map[key] = int(row['label'])  # 0 or 1

    
    instances = []
    for program in program_names:
        program_dir_path = os.path.join(static_cg_dir, program)
        program_dir_path_false = os.path.join(static_cg_dir_false, program)


        if IGNORE_LIBS:
            true_path = os.path.join(program_dir_path, 'true_filtered_libs.csv')
            false_path = os.path.join(program_dir_path_false, 'false_filtered_libs.csv')
            all_edges_path = os.path.join(program_dir_path, 'wala0cfa_filtered_libs.csv')
        else:
            true_path = os.path.join(program_dir_path, 'true_edges.csv')
            false_path = os.path.join(program_dir_path_false, 'diff_0cfa_1obj.csv')
            all_edges_path = os.path.join(program_dir_path, 'wala0cfa_filtered.csv')
        
        static_features_path = os.path.join(program_dir_path, 'static_featuers', 'wala0cfa_filtered.csv')
        
        var_features_path = os.path.join(var_features_dir, f'{program}_full_info.csv_features.csv')  #var
        trace_features_path = os.path.join(trace_features_dir, f'{program}.csv')  #cg

        if LOAD_SEMANTIC_RAW or LOAD_SEMANTIC_FINETUNED:
            semantic_features_path = os.path.join(semantic_features_dir, f'{program}.csv')

        if not os.path.exists(true_path) or not os.path.exists(all_edges_path) or not os.path.exists(static_features_path):
            continue

        true_df = pd.read_csv(true_path)
        false_df = pd.read_csv(false_path) if os.path.exists(false_path) else pd.DataFrame(columns=['method', 'offset', 'target'])
        all_df = pd.read_csv(all_edges_path)
        all_df = all_df[~all_df['method'].str.startswith('java/') & ~all_df['target'].str.startswith('java/')]

        labeled_df = pd.concat([true_df, false_df])
        labeled_keys = labeled_df[['method', 'offset', 'target']].drop_duplicates()
        unknown_df = all_df.merge(labeled_keys, on=['method', 'offset', 'target'], how='left', indicator=True)
        unknown_df = unknown_df[unknown_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        # load static features
        static_features_df = pd.read_csv(static_features_path).set_index(['method', 'offset', 'target'])

        # Load semantic features
        if LOAD_SEMANTIC_RAW or LOAD_SEMANTIC_FINETUNED:
            semantic_features_df = pd.read_csv(semantic_features_path)
            semantic_features_df = semantic_features_df.drop_duplicates(subset=['method', 'offset', 'target'])
            semantic_features_df = semantic_features_df.set_index(['method', 'offset', 'target'])
        
        # Load trace features
        if LOAD_TRACE:
            trace_features_df = pd.read_csv(trace_features_path)
            trace_features_df = trace_features_df.drop_duplicates(subset=['method', 'offset', 'target'])
            trace_features_df = trace_features_df.set_index(['method', 'offset', 'target'])

        # Load var features
        if LOAD_VAR:
            var_features_df = pd.read_csv(var_features_path)
            var_features_df = var_features_df.drop_duplicates(subset=['method', 'offset', 'target'])
            var_features_df = var_features_df.set_index(['method', 'offset', 'target'])



        def create_instance(row, label, is_unknown):
            key = (row['method'], row['offset'], row['target'])
            if key in static_features_df.index:
                inst = Instance(program, *key, is_unknown, label=label)
                static = static_features_df.loc[key]
                static = static.squeeze().values.tolist()
                inst.set_static_features(static)

                # Set trace features if available
                if LOAD_TRACE:
                    trace_key = (row['method'], row['offset'], row['target'])
                    if trace_key in trace_features_df.index:
                        trace = trace_features_df.loc[trace_key]
                        trace = trace.squeeze().values.tolist()
                        inst.set_trace_features(trace)

                # Set var features if available
                if LOAD_VAR:
                    var_key = (row['method'], row['offset'], row['target'])
                    if var_key in var_features_df.index:
                        var = var_features_df.loc[var_key]
                        var = var.squeeze().values.tolist()
                        inst.set_var_features(var)
                
                # Set semantic features if available
                if LOAD_SEMANTIC_RAW or LOAD_SEMANTIC_FINETUNED:
                    if key in semantic_features_df.index:
                        semantic = semantic_features_df.loc[key]
                        semantic = semantic.squeeze().values.tolist()
                        inst.set_semantic_features(semantic)

                # Set finetuned semantic features if available
                if LOAD_SEMANTIC_INCORRECT_FINETUNED:
                    semantic_key = (program, row['method'], row['offset'], row['target'])
                    if semantic_key in semantic_feature_map:
                        inst.set_semantic_features(semantic_feature_map[semantic_key])
      
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


