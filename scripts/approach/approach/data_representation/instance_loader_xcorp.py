from approach.data_representation.Instance import Instance
from approach.constants import static_cg_dir, features_dir, manual_labels_dir

import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict


# LOAD_SEMANTIC_FINETUNED = False
# LOAD_SEMANTIC_RAW = True


# static_cg_dir = '/20TB/mohammad/xcorpus-total-recall/dataset'
# features_dir = '/20TB/mohammad/xcorpus-total-recall/features'
# manual_labels_dir = '/20TB/mohammad/xcorpus-total-recall/manual_labeling'


# config_data = (version, config_id)
def load_instances(dataset="njr", tool=None, config_info=None, just_three=False, load_semantic_features=False, load_tokens=False, model_name=None):
    
    with open("approach/data_representation/config.json", 'r') as f:
        conf_data = json.load(f)

    if tool is None and config_info is None:
        data = conf_data.get('data', [])

    elif tool is not None and config_info is None:
        data = conf_data.get('data',[])
        data = [d for d in data if d['tool'] == tool]

    elif tool and config_info:
        data = [d for d in conf_data['data'] if d['tool'] == tool and d['config_id'] == config_info[1] and d['version'] == config_info[0]]
    
    instances = []
    for d in data:
        programs = d['programs'] if just_three else conf_data['comparison']['programs'] 
        version = d['version']
        config_id = d['config_id']
        tool = d['tool']

        
    # manual_gt_path = '/home/mohammad/projects/CallGraphPruner/data/manual/manual_unknown_labels.csv'
    
    # static_cg_dir = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'
    # static_cg_dir_false = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/static_cgs_no_exlusion'

    # if LOAD_SEMANTIC_RAW:
    #     semantic_features_dir = '/home/mohammad/projects/CallGraphPruner/data/semantic_embeddings/semantic_embeddings'

    # elif LOAD_SEMANTIC_FINETUNED:
    #     semantic_features_dir = '/home/mohammad/projects/CallGraphPruner/data/semantic_embeddings/semantic_finetuned'
    

    # with open(programs_path, 'r') as f:
    #     program_names = [line.strip() for line in f if line.strip()]

    # # Load manually labeled unknowns
    # manual_gt_map = {}
    # if os.path.exists(manual_gt_path):
    #     manual_gt_df = pd.read_csv(manual_gt_path)
    #     for _, row in manual_gt_df.iterrows():
    #         key = (row['program'], row['method'], row['offset'], row['target'])
    #         manual_gt_map[key] = int(row['label'])  # 0 or 1

    
        print(programs)
        for program in programs:
            program_dir_path = os.path.join(static_cg_dir, tool, 'without_jdk', program, f'{version}_{config_id}')
            
            true_path = os.path.join(program_dir_path, 'true_edges.csv')
            false_path = os.path.join(program_dir_path, 'false_edges.csv')
            unk_path = os.path.join(program_dir_path, 'unknown_edges.csv')
            all_edges_path = os.path.join(program_dir_path, 'total_edges.csv')

            manual_gt_path = os.path.join(manual_labels_dir, program, 'complete', 'labeling_sample.csv')
            
            static_features_path = os.path.join(features_dir, 'struct', program, f'struct_{tool}_{version}_{config_id}.csv')

            if model_name == 'codebert':
                semantic_features_path = os.path.join(features_dir, 'semantic', 'codebert', 'raw', program, f'semantic_{tool}_{version}_{config_id}.csv')
                tokens_path = os.path.join(features_dir, 'semantic', 'codebert', 'tokens', program, f'token_{tool}_{version}_{config_id}.npz')
            else:
                semantic_features_path = os.path.join(features_dir, 'semantic', 'codet5', 'raw', program, f'semantic_{tool}_{version}_{config_id}.csv')
                tokens_path = os.path.join(features_dir, 'semantic', 'codet5', 'tokens', program, f'token_{tool}_{version}_{config_id}.npz')



            if not os.path.exists(true_path) or not os.path.exists(all_edges_path) or not os.path.exists(static_features_path):
                print(f"Skipping {program} in {tool} ({version}, {config_id}) due to missing files.")
                continue

            true_df = pd.read_csv(true_path)
            false_df = pd.read_csv(false_path) if os.path.exists(false_path) else pd.DataFrame(columns=['method', 'offset', 'target'])
            unknown_df = pd.read_csv(unk_path) if os.path.exists(unk_path) else pd.DataFrame(columns=['method', 'offset', 'target'])
            all_df = pd.read_csv(all_edges_path)

            # Load manual ground truth labels
            manual_gt_map = {}
            # if os.path.exists(manual_gt_path):
                # manual_gt_df = pd.read_csv(manual_gt_path)       
                # for _, row in manual_gt_df.iterrows():
                #     key = (row['method'], row['offset'], row['target'])
                #     manual_gt_map[key] = int(row['label'])  # 0 or 1

            # load static features
            static_features_df = pd.read_csv(static_features_path).set_index(['method', 'offset', 'target'])

            # Load semantic features
            if load_semantic_features:
                semantic_features_df = pd.read_csv(semantic_features_path).set_index(['method', 'offset', 'target'])

            if load_tokens:
                data = np.load(tokens_path, allow_pickle=True)
                tokens_data = data['data'].tolist()
                tokens_df = pd.DataFrame(tokens_data).set_index(['method', 'offset', 'target'])


            # def create_instance(row, label, is_unknown):
            #     key = (row['method'], row['offset'], row['target'])
            #     if key in static_features_df.index:
            #         inst = Instance(program, *key, is_unknown, label=label)
            #         static = static_features_df.loc[key]
            #         static = static.squeeze().values.tolist()
            #         inst.set_static_features(static)
                    
            #         # Set semantic features if available
            #         if load_semantic_features:
            #             if key in semantic_features_df.index:
            #                 semantic = semantic_features_df.loc[key]
            #                 semantic = semantic.squeeze().values.tolist()
            #                 inst.set_semantic_features(semantic)
                    
            #         # Set tokens if available
            #         if load_tokens:
            #             if key in tokens_df.index:
            #                 tokens_info = tokens_df.loc[key]
            #                 tokens = tokens_info['tokens']
            #                 masks = tokens_info['masks']
            #                 inst.set_tokens(tokens)
            #                 inst.set_masks(masks)
        
            #         # Set GT if exists
            #         gt_key = (row['method'], row['offset'], row['target'])
            #         if gt_key in manual_gt_map:
            #             inst.set_ground_truth(manual_gt_map[gt_key])
                    
            #         return inst

            def create_instance(row, label, is_unknown):
                key = (row['method'], row['offset'], row['target'])

                # Check static features first
                if key not in static_features_df.index:
                    return None

                # If token loading is enabled, ensure tokens exist for the key
                if load_tokens and key not in tokens_df.index:
                    return None

                inst = Instance(program, *key, is_unknown, label=label)

                # set tool, version, and config_id
                inst.tool = tool
                inst.version = version
                inst.config_id = config_id

                # Set static features
                static = static_features_df.loc[key]
                static = static.squeeze().values.tolist()
                inst.set_static_features(static)

                # Set semantic features (optional)
                if load_semantic_features and key in semantic_features_df.index:
                    semantic = semantic_features_df.loc[key]
                    semantic = semantic.squeeze().values.tolist()
                    inst.set_semantic_features(semantic)

                # Set tokens (already verified presence)
                if load_tokens:
                    tokens_info = tokens_df.loc[key]
                    tokens = tokens_info['tokens']
                    masks = tokens_info['masks']
                    inst.set_tokens(tokens)
                    inst.set_masks(masks)

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


if __name__ == "__main__":
    # Example usage
    instances = load_instances(tool="doop", config_info=('v1', '39'), just_three=True, load_tokens=True, model_name='codebert')
    print(len(instances))
    # print first 5 instances
    for i in range(5):
        print(instances[i].get_tokens())
        print(instances[i].get_masks())

    # program_instances = defaultdict(list)
    # for inst in instances:
    #     program_instances[inst.program].append(inst)
    # program_list = list(program_instances.keys())
    # print(f"Programs: {program_list}")

    # # print len of each program's instances
    # for program, insts in program_instances.items():
    #     print(f"{program}: {len(insts)} instances")