import os
import pandas as pd

# Paths
WALA_DIR = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'
FALSE_UNK_DIR = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/static_cgs_no_exlusion'
OUTPUT_DIR = '/home/mohammad/projects/CallGraphPruner/data/datasets/no_libs'

WALA_FILTERED_NAME = 'wala0cfa_filtered_libs.csv'
WALA_FILTERED_TRUE_NAME = 'true_filtered_libs.csv'
FALSE_FILTERED_NAME = 'diff_0cfa_1obj.csv'
UNKNOWN_FILTERED_NAME = 'unknown.csv'

# Labels
LABELS = {
    'true': 1,
    'false': 0,
    'unknown': -1
}

# Process each program
for program in os.listdir(WALA_DIR):

    program_output_dir = os.path.join(OUTPUT_DIR, program)
    os.makedirs(program_output_dir, exist_ok=True)

    # Read the CSV files
    wala_df = pd.read_csv(os.path.join(WALA_DIR, program, WALA_FILTERED_NAME))
    wala_true_df = pd.read_csv(os.path.join(WALA_DIR, program, WALA_FILTERED_TRUE_NAME))
    unknown_df = pd.read_csv(os.path.join(FALSE_UNK_DIR, program, UNKNOWN_FILTERED_NAME))
    
    # Handle missing false and unknown files
    false_path = os.path.join(FALSE_UNK_DIR, program, FALSE_FILTERED_NAME)
    if os.path.exists(false_path):
        false_df = pd.read_csv(false_path)
    else:
        print(f"Warning: {FALSE_FILTERED_NAME} not found for {program}, using empty DataFrame.")
        false_df = pd.DataFrame(columns=["method", "offset", "target"])

    # Add label column
    wala_df['label'] = LABELS['unknown']  # Default to unknown
    # drop wiretap column
    wala_df.drop(columns=['wiretap'], inplace=True, errors='ignore')

    # Set true labels
    true_keys = set(wala_true_df[['method', 'offset', 'target']].apply(tuple, axis=1))
    wala_df.loc[wala_df[['method', 'offset', 'target']].apply(tuple, axis=1).isin(true_keys), 'label'] = LABELS['true']

    # Set false labels
    false_keys = set(false_df[['method', 'offset', 'target']].apply(tuple, axis=1))
    wala_df.loc[wala_df[['method', 'offset', 'target']].apply(tuple, axis=1).isin(false_keys), 'label'] = LABELS['false']

    # # Set unknown labels
    # unknown_keys = set(unknown_df[['method', 'offset', 'target']].apply(tuple, axis=1))
    # wala_df.loc[wala_df[['method', 'offset', 'target']].apply(tuple, axis=1).isin(unknown_keys), 'label'] = LABELS['unknown']

    # Save the labeled data
    output_path = os.path.join(program_output_dir, 'wala0cfa.csv')
    wala_df.to_csv(output_path, index=False)

    print(f"Processed {program} - Saved to {output_path}")
