import pandas as pd
import os
from functools import reduce

dataset_dir = "/20TB/mohammad/xcorpus-total-recall/dataset"
output_dir = '/20TB/mohammad/xcorpus-total-recall/manual_labeling2'

programs = [
    'axion',
    'batik',
    'xerces',
    'jasml'
]

tools = [
    'wala',
    'doop',
    'opal'
]

def perform_stratified_sampling(union_df, intersection_df, program, total_budget=100):
    """
    Performs stratified sampling on the union/intersection of unknown edges.

    Args:
        union_df (pd.DataFrame): DataFrame containing the union of all unknown edges.
        intersection_df (pd.DataFrame): DataFrame containing the intersection of all unknown edges.
        program (str): The name of the program being analyzed (for output file naming).
        total_budget (int): The total number of items to sample for this program.
    """
    print("\n" + "-"*20)
    print(f"Performing Stratified Sampling for {program}...")
    print(f"Total budget for labeling: {total_budget}")

    edge_columns = ['method', 'offset', 'target']

    # 1. Define the Strata: Intersection is already calculated. Now find the disagreement set.
    # The disagreement set is (Union - Intersection)
    if union_df.empty:
        print("Union DataFrame is empty, cannot perform sampling.")
        return

    # Use a merge with indicator=True to efficiently find rows in union_df but not in intersection_df
    merged = pd.merge(union_df, intersection_df, on=edge_columns, how='left', indicator=True)
    disagreement_df = merged[merged['_merge'] == 'left_only'][edge_columns].copy()

    print(f"Size of 'Disagreement' stratum: {len(disagreement_df)}")
    print(f"Size of 'Intersection' stratum: {len(intersection_df)}")

    # save the disagreement and intersection dataframes 
    program_output_dir = os.path.join(output_dir, program)
    os.makedirs(program_output_dir, exist_ok=True)
    intersection_df.to_csv(os.path.join(program_output_dir, f"intersection.csv"), index=False)
    union_df.to_csv(os.path.join(program_output_dir, f"union.csv"), index=False)
    disagreement_df.to_csv(os.path.join(program_output_dir, f"disagreement.csv"), index=False)
    return
    # 2. Allocate the Budget
    # We'll split the budget 50/50 between the two strata
    intersection_budget = total_budget // 2
    disagreement_budget = total_budget - intersection_budget # Ensures total is correct

    # 3. Sample from Each Stratum (handling cases where population is smaller than budget)
    final_samples = []

    # Sample from Intersection
    n_intersect_to_sample = min(intersection_budget, len(intersection_df))
    if n_intersect_to_sample > 0:
        intersection_sample = intersection_df.sample(n=n_intersect_to_sample, random_state=42)
        intersection_sample['source_category'] = 'intersection'
        final_samples.append(intersection_sample)

    # Sample from Disagreement
    n_disagree_to_sample = min(disagreement_budget, len(disagreement_df))
    if n_disagree_to_sample > 0:
        disagreement_sample = disagreement_df.sample(n=n_disagree_to_sample, random_state=42)
        disagreement_sample['source_category'] = 'disagreement'
        final_samples.append(disagreement_sample)

    # 4. Combine and Save
    if not final_samples:
        print("Could not draw any samples.")
        print("-" * 20 + "\n")
        return

    final_labeling_df = pd.concat(final_samples).reset_index(drop=True)

    # Save the sample to a CSV for manual labeling
    output_filename = f"labeling_sample.csv"
    final_labeling_df.to_csv(os.path.join(program_output_dir, output_filename), index=False)

    print("\n--- Sampling Results ---")
    print(f"Total edges sampled: {len(final_labeling_df)}")
    print("Distribution of samples by source:")
    print(final_labeling_df['source_category'].value_counts())
    print(f"\nLabeling sample saved to: {output_filename}")
    print("-" * 20 + "\n")


def calculate_selected_unknowns(total_unknowns_dict, program):
    selected_keys = [
        # ('wala', 'v1_4'),  #rta_full
        ('wala', 'v1_19'), #0cfa,ONE_FLOW_TO_CASTS_NO_METHOD_INVOKE (excluding xerces)
        ('wala', 'v3_0'),  #1cfa,ONE_FLOW_TO_CASTS_NO_METHOD_INVOKE (excluding xerces)
        # ('wala', 'v2_18'),  #1obj,String_only (excluding xerces)
        ('wala', 'v1_23'),  #0cfa,String_only (excluding xerces)

        ('doop', 'v1_39'),  #0cfa_on (excluding jasml)
        ('doop', 'v3_5'),   #1_type, on
        # ('doop', 'v1_3'),  # 1obj,off (excluding xerces)
        ('doop', 'v2_0'),  # 1obj,off (excluding xerces)

        ('opal', 'v1_0'),   #cha
        # ('opal', 'v1_8'),   #0-1cfa
        # ('opal', 'v1_9'),   #11cfa (excluding axion and maybe jasml )
    ]

    selected_unknowns = [v for k, v in total_unknowns_dict.items() if k in selected_keys]
    
    # Check if there are any selected dataframes
    if not selected_unknowns:
        print(f"No dataframes found for selected keys in program: {program}")
        return

    edge_columns = ['method', 'offset', 'target']

    # union of total unknowns
    union_df = pd.concat(selected_unknowns).drop_duplicates(subset=edge_columns)
    print(f"Total unknown edges for {program}: {len(union_df)}")


    # intersection of total unknowns
    # Use reduce for a more efficient and clean intersection calculation
    intersection_df = reduce(lambda left, right: pd.merge(left, right, on=edge_columns, how='inner'), selected_unknowns)
    intersection_df = intersection_df.drop_duplicates(subset=edge_columns)
    print(f"Total unknown edges intersection for {program}: {len(intersection_df)}")

    # *** NEW LOGIC: ADDED STRATIFIED SAMPLING STEP ***
    # This will generate a CSV file with samples for the current program
    perform_stratified_sampling(union_df, intersection_df, program, total_budget=100)


def main():

    for program in programs:
        total_unknowns_dict = {}

        for tool in tools:
            tool_dir = os.path.join(dataset_dir, tool, 'without_jdk', program)
            if not os.path.exists(tool_dir):
                continue

            for config in os.listdir(tool_dir):
                config_dir = os.path.join(tool_dir, config)
                if not os.path.isdir(config_dir):
                    continue

                # Read the unknowns file
                unknown_file = os.path.join(config_dir, 'unknown_edges.csv')
                if not os.path.exists(unknown_file):
                    continue

                unknown_df = pd.read_csv(unknown_file).drop_duplicates(subset=['method', 'offset', 'target'])

                if unknown_df.empty:
                    continue

                total_unknowns_dict[(tool, config)] = unknown_df

        # This single call now handles union, intersection, and sampling
        calculate_selected_unknowns(total_unknowns_dict, program)




if __name__ == '__main__':
    main()