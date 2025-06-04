
import pandas as pd
import os



tool = 'wala'
scgs_dir = f"/20TB/mohammad/xcorpus-total-recall/static_cgs/{tool}/final"
true_edges_dir = f'/20TB/mohammad/xcorpus-total-recall/dynamic_cgs'
false_edges_dir = f'/20TB/mohammad/xcorpus-total-recall/compare/{tool}/final'
unknown_edges_dir = f'/20TB/mohammad/xcorpus-total-recall/unknowns/{tool}/final'

output_dir = f'/20TB/mohammad/xcorpus-total-recall/dataset/{tool}'


app_packages = {
    'axion': ['org/axiondb',],
    'batik': ['org/apache.batik', 'org/w3c/dom'],
    'xerces': ['org/apache/xerces', 'org/apache/html/dom', 'org/apache/wml', 'org/apache/xml/serialize', 'w3c/dom/html'],
    'jasml': ['com/jasml',],
}




# def load_false_edges(program):
#     all_dfs = []
#     program_dir = os.path.join(false_edges_dir, program)
#     for file in os.listdir(program_dir):
#         if file.endswith('.csv'):
#             df = pd.read_csv(os.path.join(program_dir, file))
#             all_dfs.append(df)
#     if all_dfs:
#         return pd.concat(all_dfs, ignore_index=True).drop_duplicates()
#     else:
#         return pd.DataFrame(columns=['method', 'offset', 'target'])


# def load_unknown_edges(program):
#     all_dfs = []
#     program_dir = os.path.join(scgs_dir, program)
#     for file in os.listdir(program_dir):
#         if file.endswith('.csv'):
#             df = pd.read_csv(os.path.join(program_dir, file))
#             all_dfs.append(df)
#     if all_dfs:
#         return pd.concat(all_dfs, ignore_index=True).drop_duplicates()
#     else:
#         return pd.DataFrame(columns=['method', 'offset', 'target'])
    
# def load_scgs(program):
#     all_dfs = []
#     program_dir = os.path.join(scgs_dir, program)
#     for file in os.listdir(program_dir):
#         if file.endswith('.csv'):
#             df = pd.read_csv(os.path.join(program_dir, file))
#             all_dfs.append(df)
#     if all_dfs:
#         return pd.concat(all_dfs, ignore_index=True).drop_duplicates()
#     else:
#         return pd.DataFrame(columns=['method', 'offset', 'target'])



def remove_jdk_edges(program, df):
    app_packages_list = app_packages.get(program, [])

    mask = (
        df['method'].str.startswith(tuple(app_packages_list)) |
        df['target'].str.startswith(tuple(app_packages_list))
    )
    return df[mask].reset_index(drop=True).drop_duplicates()


# def main():
    
#     for program in os.listdir(scgs_dir):
#         print(f"Processing program: {program}")
#         true_df = pd.read_csv(os.path.join(true_edges_dir, program, 'dcg.csv')).drop_duplicates()
#         false_df = load_false_edges(program)
#         unknown_df = load_unknown_edges(program)
#         # total_df = pd.concat([true_df, false_df, unknown_df], ignore_index=True).drop_duplicates()
#         total_df = load_scgs(program)

#         key_cols = ['method', 'offset', 'target']  # adjust as needed

#         # Merge to find rows in true_df that also exist in unknown_df
#         overlap = true_df.merge(unknown_df[key_cols], on=key_cols, how='inner')

#         if not overlap.empty:
#             print(f"Found {len(overlap)} rows in true that are also in unknown for program {program}")
#             print(overlap)
#         else:
#             print(f"No overlap found for {program}")

#         # stats with jdk edges
#         print(f"Stats with JDK edges for {program}:")
#         print(f"Total edges: {len(total_df)}")
#         print(f"True edges: {len(true_df)}")
#         print(f"False edges: {len(false_df)}")
#         print(f"Unknown edges1: {len(total_df) - len(true_df) - len(false_df)}")
#         print(f"Unknown edges2: {len(unknown_df)}")

#         print("******************************************")

#         f_total = remove_jdk_edges(program, total_df)
#         f_true = remove_jdk_edges(program, true_df)
#         f_false = remove_jdk_edges(program, false_df)
#         f_unknown = remove_jdk_edges(program, unknown_df)

#         # stats without jdk edges
#         print(f"Stats without JDK edges for {program}:")
#         print(f"Total edges: {len(f_total)}")
#         print(f"True edges: {len(f_true)}")
#         print(f"False edges: {len(f_false)}")
#         print(f"Unknown edges: {len(f_total) - len(f_true) - len(f_false)}")
#         print(f"Unknown edges2: {len(f_unknown)}")

#         print("******************************************")


def get_false_edges(program_name, version, config_id):
    """union of all false edges for a specific program with a specific configuration"""

    false_program_dir = os.path.join(false_edges_dir, program_name)

    all_false_edges = []
    for file in os.listdir(false_program_dir):
        if file.startswith(f"{version}_{config_id}_"):
            false_edges_path = os.path.join(false_program_dir, file)
            false_df = pd.read_csv(false_edges_path)
            if false_df is not None:
                false_df = false_df.drop_duplicates(subset=['method', 'offset', 'target'])
                all_false_edges.append(false_df)
    if all_false_edges:
        all_false_edges_df = pd.concat(all_false_edges, ignore_index=True).drop_duplicates(['method', 'offset', 'target'])
        return all_false_edges_df
    return pd.DataFrame(columns=['method', 'offset', 'target'])



def get_true_edges(total_df, program):
    true_path = os.path.join(true_edges_dir, program, 'dcg.csv')
    
    if not os.path.exists(true_path):
        return pd.DataFrame(columns=total_df.columns)  # or handle it as needed

    true_df = pd.read_csv(true_path).drop_duplicates(subset=['method', 'offset', 'target'])

    # Convert to tuple for matching
    total_tuples = total_df[['method', 'offset', 'target']].apply(tuple, axis=1)
    true_tuples = set(true_df[['method', 'offset', 'target']].apply(tuple, axis=1))

    return total_df[total_tuples.isin(true_tuples)].reset_index(drop=True).drop_duplicates()


def get_unknown_edges(total_df, false_df, true_df):

    """Get unknown edges by removing true and false edges from total edges"""
    
    # Convert to tuple for matching
    total_tuples = total_df[['method', 'offset', 'target']].apply(tuple, axis=1)
    true_tuples = set(true_df[['method', 'offset', 'target']].apply(tuple, axis=1))
    false_tuples = set(false_df[['method', 'offset', 'target']].apply(tuple, axis=1))

    unknown_mask = ~total_tuples.isin(true_tuples) & ~total_tuples.isin(false_tuples)
    
    return total_df[unknown_mask].reset_index(drop=True).drop_duplicates()



def main():

    for program in os.listdir(scgs_dir):
        program_dir = os.path.join(scgs_dir, program)
        for scg_name in os.listdir(program_dir):
            parts = scg_name.split('_')
            version = parts[1]
            config_id = parts[2].split('.')[0]

            total_df = pd.read_csv(os.path.join(program_dir, scg_name)).drop_duplicates()
            true_df = get_true_edges(total_df, program)
            false_df = get_false_edges(program, version, config_id)
            # unknown_df = pd.read_csv(os.path.join(unknown_edges_dir, program, f"unknown_{version}_{config_id}.csv")).drop_duplicates()
            unknown_df = get_unknown_edges(total_df, false_df, true_df)


            with_jdk_output_dir = os.path.join(output_dir, 'with_jdk', program, f'{version}_{config_id}')
            os.makedirs(with_jdk_output_dir, exist_ok=True)
            total_df.to_csv(os.path.join(with_jdk_output_dir, 'total_edges.csv'), index=False)
            true_df.to_csv(os.path.join(with_jdk_output_dir, 'true_edges.csv'), index=False)
            false_df.to_csv(os.path.join(with_jdk_output_dir, 'false_edges.csv'), index=False)
            unknown_df.to_csv(os.path.join(with_jdk_output_dir, 'unknown_edges.csv'), index=False)

            stats = {
                'program': program,
                'version': version,
                'config_id': config_id,
                'total_edges': len(total_df),
                'true_edges': len(true_df),
                'false_edges': len(false_df),
                'unknown_edges': len(unknown_df)
            } 

            stats_df = pd.DataFrame([stats])
            stats_file = os.path.join(output_dir, 'with_jdk', f'{tool}_with_jdk_stats.csv')
            if os.path.exists(stats_file):
                stats_df.to_csv(stats_file, mode='a', header=False, index=False)
            else:
                stats_df.to_csv(stats_file, index=False)


            # Now remove JDK edges
            f_total = remove_jdk_edges(program, total_df)
            f_true = remove_jdk_edges(program, true_df)
            f_false = remove_jdk_edges(program, false_df)
            f_unknown = remove_jdk_edges(program, unknown_df)
            without_jdk_output_dir = os.path.join(output_dir, 'without_jdk', program, f'{version}_{config_id}')
            os.makedirs(without_jdk_output_dir, exist_ok=True)
            f_total.to_csv(os.path.join(without_jdk_output_dir, 'total_edges.csv'), index=False)
            f_true.to_csv(os.path.join(without_jdk_output_dir, 'true_edges.csv'), index=False)
            f_false.to_csv(os.path.join(without_jdk_output_dir, 'false_edges.csv'), index=False)
            f_unknown.to_csv(os.path.join(without_jdk_output_dir, 'unknown_edges.csv'), index=False)
            f_stats = {
                'program': program,
                'version': version,
                'config_id': config_id,
                'total_edges': len(f_total),
                'true_edges': len(f_true),
                'false_edges': len(f_false),
                'unknown_edges': len(f_unknown)
            }
            f_stats_df = pd.DataFrame([f_stats])
            f_stats_file = os.path.join(output_dir, 'without_jdk', f'{tool}_without_jdk_stats.csv')
            if os.path.exists(f_stats_file):
                f_stats_df.to_csv(f_stats_file, mode='a', header=False, index=False)
            else:
                f_stats_df.to_csv(f_stats_file, index=False)
            print(f"Processed {program} - {version} - {config_id}")


if __name__ == "__main__":
    main()