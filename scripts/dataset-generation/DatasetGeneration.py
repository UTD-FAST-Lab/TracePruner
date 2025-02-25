


import os, re
import json
import pandas as pd


class DatasetGeneration:
    def __init__(self, programs_list_file, njr1_dir, encoded_edge_traces_dir, output_file):
        """
        :param programs_list_file: Path to a file listing program directories (one per line).
        :param output_file: Path where the combined dataset CSV will be saved.
        """
        self.programs_list_file = programs_list_file
        self.output_file = output_file
        self.njr1_dir = njr1_dir
        self.encoded_edge_traces_dir = encoded_edge_traces_dir


    def load_njr_file(self, program_name):
        '''load njr file that has the labels'''

        program_dir = os.path.join(self.njr1_dir, program_name)
        njr1_file = os.path.join(program_dir, 'wala0cfa.csv')

        if not os.path.exists(njr1_file):
            print(f"Missing njr file in {program_dir}. Skipping.")
            return None
        
        # Load the file
        try:
            njr_df = pd.read_csv(njr1_file)
        except Exception as e:
            print(f"Error reading njr file in {program_dir}: {e}")
            return None

        return njr_df



    def load_trace_feature_file(self, program_name):
        '''loads features of each program along with its mapping json'''

        program_dir = os.path.join(self.encoded_edge_traces_dir, program_name)

        mapping_file = os.path.join(program_dir, "segments.json")
        features_file = os.path.join(program_dir, "w2v_features.csv")

        
        if not os.path.exists(mapping_file):
            print(f"Missing mapping file in {program_dir}. Skipping.")
            return None
        if not os.path.exists(features_file):
            print(f"Missing features file in {program_dir}. Skipping.")
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
        
        return (mapping_data, features_df)
    
    
    def reformat_node_string(self, node_string):
        pattern = r"Node: < (?:Primordial|Application), ([^,]+), ([^ ]+) >"
        match = re.search(pattern, node_string)
        
        if match:
            class_name = match.group(1)
            method_signature = match.group(2)
            
            # Remove the leading 'L' from the class name if present
            class_name = class_name.lstrip('L')
            
            # Replace the space before method signature with ':'
            formatted_string = f"{class_name}.{method_signature.split('(')[0]}:({method_signature.split('(')[1]}"
            return formatted_string
        return None  # Return None if the format is incorrect


    def process_program(self, program_name):
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
        

        njr_df = self.load_njr_file(program_name)
        mapping_data, features_df = self.load_trace_feature_file(program_name)

       

        # Ensure there is a 'label' column in features; if not, add it.
        if 'label' not in features_df.columns:
            features_df['label'] = 'NA'

        # Create a list to accumulate matched NJR rows for this program.
        matched_njr_records = []

        # Check if features.csv has an "edge_name" column to match mapping records
        if 'edge_name' in features_df.columns:
            # Build a lookup from edge_path to mapping record
            mapping_lookup = {f"{record['edge_name']}.log": record for record in mapping_data}
            for idx, row in features_df.iterrows():
                edge_name = row['edge_name']
                if edge_name in mapping_lookup:
                    record = mapping_lookup[edge_name]
                    src = record.get('src')
                    target = record.get('target')
                else:
                    src, target = None, None

                if src and target:
                    formatted_src = self.reformat_node_string(src)
                    formatted_target = self.reformat_node_string(target)
                    # Look up the corresponding edge in the njr dataframe.
                    match = njr_df[(njr_df['method'] == formatted_src) & (njr_df['target'] == formatted_target)]
                    if not match.empty:
                        # Use the first match's label and record.
                        best_match = match.iloc[0]
                        # Use the first match's label.
                        label = match.iloc[0].get('wiretap', 'NA')
                        features_df.at[idx, 'label'] = label
                        features_df.at[idx, 'method'] = formatted_src
                        features_df.at[idx, 'target'] = formatted_target
                        matched_njr_records.append(best_match.to_dict())
                    else:
                        features_df.at[idx, 'label'] = 'NA'
                else:
                    features_df.at[idx, 'label'] = 'NA'

        program_dir = os.path.join(self.encoded_edge_traces_dir, program_name)
        updated_features_file = os.path.join(program_dir, "w2v_features_labels.csv")
        try:
            features_df.to_csv(updated_features_file, index=False)
  
            # print(f"Processed {program_dir} — updated features saved to {updated_features_file}")
        except Exception as e:
            print(f"Error saving updated features in {program_dir}: {e}")


        # Convert the matched records list to a DataFrame (one row per feature row)
        matched_njr_df = pd.DataFrame(matched_njr_records)

        # Return both the updated features and the matched NJR rows.
        return features_df, matched_njr_df

    def generate_dataset(self):
        """
        Process every program directory listed in programs.txt and combine all updated
        features into a single CSV dataset.
        """
        all_features = []
        all_matched_njr = []

        for program_name in os.listdir(self.encoded_edge_traces_dir):
            features_df, program_matched_df = self.process_program(program_name)
            if features_df is not None:
                # Optionally add a column indicating the program (using the folder name)
                features_df['program_name'] = program_name
                all_features.append(features_df)

            if not program_matched_df.empty:
                # Also mark the program for matched NJR rows if desired.
                program_matched_df['program_name'] = program_name
                all_matched_njr.append(program_matched_df)

        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            try:
                combined_df.to_csv(self.output_file, index=False)
                print(f"Combined dataset saved to {self.output_file}")
            except Exception as e:
                print(f"Error saving combined dataset: {e}")
        else:
            print("No programs processed successfully; dataset not generated.")

        
        # Combine all matched NJR rows and save them.
        if all_matched_njr:
            combined_matched_df = pd.concat(all_matched_njr, ignore_index=True)
            # For example, you can name the file similarly but with a different suffix.
            matched_output_file = self.output_file.replace(".csv", "_cgpruner.csv")
            try:
                combined_matched_df.to_csv(matched_output_file, index=False)
                print(f"Combined matched NJR dataset saved to {matched_output_file}")
            except Exception as e:
                print(f"Error saving combined matched NJR dataset: {e}")
        else:
            print("No matched NJR rows were found; matched dataset not generated.")


if __name__ == '__main__':

    data_folder = '/home/mohammad/projects/CallGraphPruner_data'  
    NJR1_DATASET_FOLDER = f'{data_folder}/dataset-high-precision-callgraphs/full_callgraphs_set'
    PROGRAM_FILES = '/home/mohammad/projects/CallGraphPruner/scripts/trace-generation/programs.txt'

    encoded_edge_traces_dir = '/home/mohammad/projects/CallGraphPruner/data/encoded-edge'


    dataset_path = '/home/mohammad/projects/CallGraphPruner/data/datasets'
    output_file = os.path.join(dataset_path, 'combined_w2v_dataset_v0.csv')

    # Create a DatasetGeneration instance and generate the dataset.
    dataset_gen = DatasetGeneration(programs_list_file=PROGRAM_FILES, njr1_dir=NJR1_DATASET_FOLDER, encoded_edge_traces_dir=encoded_edge_traces_dir, output_file=output_file)
    dataset_gen.generate_dataset()
