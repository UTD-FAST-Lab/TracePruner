import os
import pandas as pd


config_file_version = "v1"
tool_name = "wala"
scgs_dir = f"/20TB/mohammad/xcorpus-total-recall/static_cgs/{tool_name}/{config_file_version}"
false_dir = f"/20TB/mohammad/xcorpus-total-recall/compare/{tool_name}/{config_file_version}"
true_dir = "/20TB/mohammad/xcorpus-total-recall/dynamic_cgs"
unk_dir = f"/20TB/mohammad/xcorpus-total-recall/unknowns/{tool_name}/{config_file_version}"

def get_true_edges(program_name):
    true_program_path = os.path.join(true_dir, program_name, 'dcg.csv')
    true_df = pd.read_csv(true_program_path)
    if true_df is not None:
        true_df = true_df.drop_duplicates(subset=['method', 'offset', 'target'])
        return true_df
    return None

def get_false_edges(program_name, config_id):
    """union of all false edges for a specific program with a specific configuration"""

    false_program_dir = os.path.join(false_dir, program_name)

    all_false_edges = []
    for file in os.listdir(false_program_dir):
        if file.startswith(config_id):
            false_edges_path = os.path.join(false_program_dir, file)
            false_df = pd.read_csv(false_edges_path)
            all_false_edges.append(false_df)
    if all_false_edges:
        all_false_edges_df = pd.concat(all_false_edges, ignore_index=True).drop_duplicates()
        return all_false_edges_df
    return None

def get_all_edges(program_name, config_id):
    """union of all edges for a specific program with a specific configuration"""
    scg_path = os.path.join(scgs_dir, program_name, 'wala_{}.csv'.format(config_id))
    if os.path.exists(scg_path):
        scg_df = pd.read_csv(scg_path)
        if scg_df is not None:
            scg_df = scg_df.drop_duplicates(subset=['method', 'offset', 'target'])
            return scg_df
    return None

def save_unknowns(program_name, config_id, unknown_df):
    
    unknown_path = os.path.join(unk_dir, program_name, f"unknown_{config_id}.csv")
    os.makedirs(os.path.dirname(unknown_path), exist_ok=True)
    unknown_df.to_csv(unknown_path, index=False)

def main(program_name, config_id):

    true_df = get_true_edges(program_name)

    false_df = get_false_edges(program_name, config_id)

    all_df = get_all_edges(program_name, config_id)

    if true_df is None or false_df is None or all_df is None:
        print(f"Missing data for program {program_name} with config {config_id}.")
        return
    all_df = all_df[~all_df['method'].str.startswith('<boot>')]
    labeled_df = pd.concat([true_df, false_df])


    if labeled_df.duplicated(subset=['method', 'offset', 'target']).any():
        # print the duplicate rows
        duplicates = labeled_df[labeled_df.duplicated(subset=['method', 'offset', 'target'], keep=False)]
        print("Duplicate edges found in labeled data:")
        print(duplicates.shape)
        # raise ValueError("Duplicate edges found in labeled data.")
    
    labeled_df = labeled_df.drop_duplicates(subset=['method', 'offset', 'target'])
    labeled_keys = labeled_df[['method', 'offset', 'target']]
    unknown_df = all_df.merge(labeled_keys, on=['method', 'offset', 'target'], how='left', indicator=True)
    unknown_df = unknown_df[unknown_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    # unknown_df = all_df.merge(true_df, on=['method', 'offset', 'target'], how='left', indicator=True)
    # unknown_df = unknown_df[unknown_df['_merge'] == 'both'].drop(columns=['_merge'])



    save_unknowns(program_name, config_id, unknown_df)


if __name__ == "__main__":
    program_name = "xerces"
    config_id = "9"  
    main(program_name, config_id)