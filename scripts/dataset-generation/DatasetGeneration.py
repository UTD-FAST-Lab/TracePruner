


import os
import json
import pandas as pd

class DatasetGeneration:
    def __init__(self, programs_list_file='programs.txt', output_file='combined_dataset.csv'):
        """
        :param programs_list_file: Path to a file listing program directories (one per line).
        :param output_file: Path where the combined dataset CSV will be saved.
        """
        self.programs_list_file = programs_list_file
        self.output_file = output_file
        self.program_dirs = self.load_programs()

    def load_programs(self):
        """Read the programs.txt file and return a list of program directory paths."""
        program_dirs = []
        with open(self.programs_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    program_dirs.append(line)
        return program_dirs

    def process_program(self, program_dir):
        """
        Process one program's files:
          - Read njr CSV file ("njr-1 wala0.csv")
          - Read edge mapping file ("edge_map.json")
          - Read features CSV file ("features.csv")
          - For each row in features, find the corresponding mapping to extract src/target,
            then search the njr file for the matching edge and obtain its label.
          - Save an updated features CSV (named features_updated.csv) in the program's folder.
        :param program_dir: The directory path for the program.
        :return: A pandas DataFrame with the updated features.
        """
        njr_file = os.path.join(program_dir, "njr-1 wala0.csv")
        mapping_file = os.path.join(program_dir, "edge_map.json")
        features_file = os.path.join(program_dir, "features.csv")

        # Check that all required files exist.
        if not os.path.exists(njr_file):
            print(f"Missing njr file in {program_dir}. Skipping.")
            return None
        if not os.path.exists(mapping_file):
            print(f"Missing mapping file in {program_dir}. Skipping.")
            return None
        if not os.path.exists(features_file):
            print(f"Missing features file in {program_dir}. Skipping.")
            return None

        # Load the files
        try:
            njr_df = pd.read_csv(njr_file)
        except Exception as e:
            print(f"Error reading njr file in {program_dir}: {e}")
            return None

        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
        except Exception as e:
            print(f"Error reading mapping file in {program_dir}: {e}")
            return None

        try:
            features_df = pd.read_csv(features_file)
        except Exception as e:
            print(f"Error reading features file in {program_dir}: {e}")
            return None

        # Ensure there is a 'label' column in features; if not, add it.
        if 'label' not in features_df.columns:
            features_df['label'] = 'NA'

        # Check if features.csv has an "edge_path" column to match mapping records
        if 'edge_path' in features_df.columns:
            # Build a lookup from edge_path to mapping record
            mapping_lookup = {record['edge_path']: record for record in mapping_data}
            for idx, row in features_df.iterrows():
                edge_path = row['edge_path']
                if edge_path in mapping_lookup:
                    record = mapping_lookup[edge_path]
                    src = record.get('src')
                    target = record.get('target')
                else:
                    src, target = None, None

                if src and target:
                    # Look up the corresponding edge in the njr dataframe.
                    match = njr_df[(njr_df['src'] == src) & (njr_df['target'] == target)]
                    if not match.empty:
                        # Use the first match's label.
                        label = match.iloc[0].get('label', 'NA')
                        features_df.at[idx, 'label'] = label
                    else:
                        features_df.at[idx, 'label'] = 'NA'
                else:
                    features_df.at[idx, 'label'] = 'NA'
        else:
            # If there's no "edge_path", assume the order of mapping_data aligns with features_df.
            for idx, row in features_df.iterrows():
                if idx < len(mapping_data):
                    record = mapping_data[idx]
                    src = record.get('src')
                    target = record.get('target')
                    if src and target:
                        match = njr_df[(njr_df['src'] == src) & (njr_df['target'] == target)]
                        if not match.empty:
                            label = match.iloc[0].get('label', 'NA')
                            features_df.at[idx, 'label'] = label
                        else:
                            features_df.at[idx, 'label'] = 'NA'
                    else:
                        features_df.at[idx, 'label'] = 'NA'
                else:
                    features_df.at[idx, 'label'] = 'NA'

        # Optionally, save the updated features CSV in the program folder.
        updated_features_file = os.path.join(program_dir, "features_updated.csv")
        try:
            features_df.to_csv(updated_features_file, index=False)
            print(f"Processed {program_dir} — updated features saved to {updated_features_file}")
        except Exception as e:
            print(f"Error saving updated features in {program_dir}: {e}")

        return features_df

    def generate_dataset(self):
        """
        Process every program directory listed in programs.txt and combine all updated
        features into a single CSV dataset.
        """
        all_features = []
        for program_dir in self.program_dirs:
            features_df = self.process_program(program_dir)
            if features_df is not None:
                # Optionally add a column indicating the program (using the folder name)
                features_df['program'] = os.path.basename(os.path.normpath(program_dir))
                all_features.append(features_df)

        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            try:
                combined_df.to_csv(self.output_file, index=False)
                print(f"Combined dataset saved to {self.output_file}")
            except Exception as e:
                print(f"Error saving combined dataset: {e}")
        else:
            print("No programs processed successfully; dataset not generated.")


if __name__ == '__main__':
    # Create a DatasetGeneration instance and generate the dataset.
    dataset_gen = DatasetGeneration(programs_list_file='programs.txt', output_file='combined_dataset.csv')
    dataset_gen.generate_dataset()












    #for each program in the the encoded in programs.txt
        # find and read njr-1 wala0.csv
        # find and read the json mapping of edges
        # find and read the features.csv file of the program
        # for each row in features.csv:
            # get the src and target from searching the mapping json file
            # find the edge in njr file by mapping src and target
            # get the label and add it to the label column of features.csv file
                # if the label exists, put label if not put NA ( for further inverstigation)

    # collect all of the features.csv files from the programs' folders and combine them into a single csv file as the dataset

   