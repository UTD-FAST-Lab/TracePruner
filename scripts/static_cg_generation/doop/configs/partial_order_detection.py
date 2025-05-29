

static_cgs_dir = '/20TB/mohammad/xcorpus-total-recall/static_cgs/doop'
config_csv = "/home/mohammad/projects/TracePruner/scripts/static_cg_generation/doop/configs/doop_default_1change_configs_v1.csv"
config_file_version = "v1"
json_file = "/home/mohammad/projects/TracePruner/scripts/static_cg_generation/doop/configs/doop_config.json"

output_dir = '/20TB/mohammad/xcorpus-total-recall/compare'

programs = [
    'axion',
    'jasml',
    'batik',
    'xerces'
]


import os
import pandas as pd
import json
import numpy as np



def has_partial_order(orders, analysis1, analysis2):
    """
    Check if the two analyses have a partial order.
    """

    for order in orders:
        if order["left"] == analysis2 and order["right"] == analysis1:
            # print(f"Found partial order: {analysis2} < {analysis1}")
            return True
        
    return False


def find_diff(config1_id, config2_id):
    """
    Find the differences between the static call graphs of two configurations.
    """

    scgs_dir = os.path.join(static_cgs_dir, config_file_version, 'reformatted')

    for program in programs:

        scg1_path = os.path.join(scgs_dir, f'{program}_{config1_id}', 'CallGraphEdge.csv')
        
        if not os.path.exists(scg1_path):
            # print(f"Static call graph for config {config1_id} not found in {program}. Skipping.")
            continue
        scg2_path = os.path.join(scgs_dir, f'{program}_{config2_id}', 'CallGraphEdge.csv')
        if not os.path.exists(scg2_path):
            # print(f"Static call graph for config {config2_id} not found in {program}. Skipping.")
            continue
        # Read the static call graphs
        scg1 = pd.read_csv(scg1_path)
        scg2 = pd.read_csv(scg2_path)
       

        if scg1 is not None and scg2 is not None:
            # datapoints in scg1 but not in scg2
            # Ensure the columns exist
            key_cols = ["method", "offset", "target"]

            # Filter rows in df1 that are not in df2 on those columns
            diff_df = scg1.merge(scg2[key_cols], on=key_cols, how='left', indicator=True)
            df_only_in_df1 = diff_df[diff_df['_merge'] == 'left_only'].drop(columns=['_merge'])

            if not df_only_in_df1.empty:
                write_diff_to_csv(program, config1_id, config2_id, df_only_in_df1)
        

def write_diff_to_csv(program, config1_id, config2_id, diff):
    """
    Write the differences to a CSV file.
    """
    diff_dir = os.path.join(output_dir, 'doop', config_file_version, program)
    os.makedirs(diff_dir, exist_ok=True)

    # Create a filename based on the configuration IDs and version
    diff_filename = f"{config1_id}_{config2_id}_diff.csv"
    diff_file = os.path.join(diff_dir, diff_filename)
    diff.to_csv(diff_file, index=False)


def configs_match_except_analysis(config, config_compare):
    for key in config:
        if key == 'analysis':
            continue
        if config.get(key) != config_compare.get(key):
            return False
    return True


def main():
    configs = pd.read_csv(config_csv).to_dict(orient="records")
    # json_partial = pd.read_json(json_file, orient="records").to_dict(orient="records")
    with open(json_file, 'r') as f:
        json_partial = json.load(f)
    for option in json_partial['options']:
        if option['name'] == 'analysis':
            orders = option['orders']
            break

    diff_pairs = []
    for i, config in enumerate(configs):
        for j, config_compare in enumerate(configs):
            if config == config_compare:
                continue
            if configs_match_except_analysis(config, config_compare):
                if has_partial_order(orders, config["analysis"], config_compare["analysis"]):
                    diff_pairs.append((i, j))

    print(f"Found {len(diff_pairs)} pairs with partial order.")

    for config1_id, config2_id in diff_pairs:
        find_diff(config1_id, config2_id)


if __name__ == "__main__":
    main()