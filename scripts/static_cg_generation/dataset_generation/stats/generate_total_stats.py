import os
import pandas as pd
from collections import defaultdict


auto_path = '/20TB/mohammad/xcorpus-total-recall/dataset'


def total_results():

    true_list = []
    false_list = []
    unknown_list = []
    total_list = []

    for tool in os.listdir(auto_path):
        wo_path = os.path.join(auto_path, tool, 'without_jdk')

        for program in os.listdir(wo_path):
            program_path = os.path.join(wo_path, program)
            if not os.path.isdir(program_path):
                continue

            for config in os.listdir(program_path):
                config_path = os.path.join(program_path, config)

                true_df = pd.read_csv(os.path.join(config_path, 'true_edges.csv'))
                false_df = pd.read_csv(os.path.join(config_path, 'false_edges.csv'))
                unknown_df = pd.read_csv(os.path.join(config_path, 'unknown_edges.csv'))
                total_df = pd.concat([true_df, false_df, unknown_df], ignore_index=True)

                true_list.append(true_df)
                false_list.append(false_df)
                unknown_list.append(unknown_df)
                total_list.append(total_df)

    # Concatenate all DataFrames in each list
    true_all = pd.concat(true_list, ignore_index=True)
    false_all = pd.concat(false_list, ignore_index=True)
    unknown_all = pd.concat(unknown_list, ignore_index=True)
    total_all = pd.concat(total_list, ignore_index=True)

    # Drop duplicates based on (method, offset, target)
    true_all = true_all.drop_duplicates(subset=['method', 'offset', 'target'])
    false_all = false_all.drop_duplicates(subset=['method', 'offset', 'target'])
    unknown_all = unknown_all.drop_duplicates(subset=['method', 'offset', 'target'])
    total_all = total_all.drop_duplicates(subset=['method', 'offset', 'target'])

    # Print statistics
    print("✅ Unique edge counts after deduplication:")
    print(f"True edges:     {len(true_all)}")
    print(f"False edges:    {len(false_all)}")
    print(f"Unknown edges:  {len(unknown_all)}")
    print(f"Total edges:    {len(total_all)}")


def per_program_results():
    true_list = defaultdict(list)
    false_list = defaultdict(list)
    unknown_list = defaultdict(list)
    total_list = defaultdict(list)

    for tool in os.listdir(auto_path):
        wo_path = os.path.join(auto_path, tool, 'without_jdk')

        for program in os.listdir(wo_path):
            program_path = os.path.join(wo_path, program)
            if not os.path.isdir(program_path):
                continue

            for config in os.listdir(program_path):
                config_path = os.path.join(program_path, config)

                true_df = pd.read_csv(os.path.join(config_path, 'true_edges.csv'))
                false_df = pd.read_csv(os.path.join(config_path, 'false_edges.csv'))
                unknown_df = pd.read_csv(os.path.join(config_path, 'unknown_edges.csv'))
                total_df = pd.concat([true_df, false_df, unknown_df], ignore_index=True)

                true_list[program].append(true_df)
                false_list[program].append(false_df)
                unknown_list[program].append(unknown_df)
                total_list[program].append(total_df)


    for program in true_list.keys():
        print(f"Processing program: {program}")

        # Concatenate all DataFrames for the current program
        true_all = pd.concat(true_list[program], ignore_index=True)
        false_all = pd.concat(false_list[program], ignore_index=True)
        unknown_all = pd.concat(unknown_list[program], ignore_index=True)
        total_all = pd.concat(total_list[program], ignore_index=True)

        # Drop duplicates based on (method, offset, target)
        true_all = true_all.drop_duplicates(subset=['method', 'offset', 'target'])
        false_all = false_all.drop_duplicates(subset=['method', 'offset', 'target'])
        unknown_all = unknown_all.drop_duplicates(subset=['method', 'offset', 'target'])
        total_all = total_all.drop_duplicates(subset=['method', 'offset', 'target'])

        # Print statistics for the current program
        print(f"✅ Unique edge counts after deduplication for {program}:")
        print(f"True edges:     {len(true_all)}")
        print(f"False edges:    {len(false_all)}")
        print(f"Unknown edges:  {len(unknown_all)}")
        print(f"Total edges:    {len(total_all)}")
        print()

def main(total=False):

    if total:
        total_results()
    else:
        per_program_results()





if __name__ == "__main__":
    total = False
    main(total)