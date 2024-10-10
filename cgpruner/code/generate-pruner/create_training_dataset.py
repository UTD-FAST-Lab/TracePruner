import os
import pandas as pd
from random import sample

data_path = f'/20TB/mohammad'
model = 'original'   #config , config_trace, original

# Paths to directories and scripts
TRAINING_CGS_DIR = f'{data_path}/models/pruner_{model}/output/static_cgs/training_cgs'
DYNAMIC_CGS_DIR = f'{data_path}/dataset-high-precision-callgraphs/full_callgraphs_set'
ADD_FEATURE_SCRIPT = '/home/mohammad/projects/cgPruner/CallGraphPruner/cgpruner/code/run-on-single-program/tools/balancer-cg/generate_extra_information_from_dataset.py'
TRAINING_DATA_DIR = f'{data_path}/models/pruner_{model}/training_data'

# Function to add features using an external script
def add_features(static_cg, output_file):
    os.system(f'python3 {ADD_FEATURE_SCRIPT} {static_cg} {output_file}')


# Function to add wiretap labels to the training data from dynamic CGs
def add_wiretap_labels(dynamic_cg, training_data):
    # Read the dynamic CG file
    dynamic_df = pd.read_csv(dynamic_cg)
    
    # Filter rows in dynamic_df where wiretap label is 1 (this is dynamicCG)
    dynamicCG = dynamic_df[dynamic_df['wiretap'] == 1][['method', 'target']]

    # Initialize the 'wiretap' column in training_data with default value 0
    training_data['wiretap'] = 0

    # Iterate through each row in training_data
    for idx, row in training_data.iterrows():
        # Check if the method-target pair exists in dynamicCG
        if ((dynamicCG['method'] == row['method']) & (dynamicCG['target'] == row['target'])).any():
            # If a match is found, set wiretap to 1
            training_data.at[idx, 'wiretap'] = 1

    # Return the modified training_data
    return training_data


# Main function to process all programs
def main():
    all_training_data = []

    # Loop over each program directory in training_cgs folder
    for program in os.listdir(TRAINING_CGS_DIR):
        program_dir = os.path.join(TRAINING_CGS_DIR, program)

        if not os.path.isdir(program_dir):
            continue

        program_training_data = []

        # Loop over each configuration file (.csv) in the program folder
        for config_file in os.listdir(program_dir):
            if config_file.endswith('.csv'):
                static_cg = os.path.join(program_dir, config_file)
                
                # Run the add feature script and save the files in training_data folder
                training_output_file = os.path.join(TRAINING_DATA_DIR, program, f"struct_{config_file}")
                os.makedirs(os.path.dirname(training_output_file), exist_ok=True)
                add_features(static_cg, training_output_file)

                # Read the generated training data
                training_data = pd.read_csv(training_output_file)

                # Read the dynamic CG and add wiretap labels to the training data
                dynamic_cg_file = os.path.join(DYNAMIC_CGS_DIR, program, f"wala0cfa.csv")
                if os.path.exists(dynamic_cg_file):
                    training_data = add_wiretap_labels(dynamic_cg_file, training_data)

                # Append the training data for this config
                program_training_data.append(training_data)

        # Concatenate all training dataframes for the program, and drop duplicates
        if program_training_data:
            program_df = pd.concat(program_training_data).drop_duplicates()

            # If more than 20k datapoints, subsample to 20k
            if len(program_df) > 20000:
                program_df = program_df.sample(n=20000, random_state=42)

            # Drop any unnamed columns (those without a header)
            program_df = program_df.loc[:, ~program_df.columns.str.contains('^Unnamed')]
            program_df.to_csv(os.path.join(TRAINING_DATA_DIR, program, 'subsampled.csv'))

            all_training_data.append(program_df)

    # Save the final concatenated training data
    if all_training_data:
        final_df = pd.concat(all_training_data).drop_duplicates()
        final_df.to_csv(os.path.join(TRAINING_DATA_DIR,'final_training_data.csv'), index=False)

if __name__ == '__main__':
    main()
