import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import re
import json

class LabelExtraction:


    def __init__(self, njr1_dir, edge_traces_dir):
        self.njr1_dir = njr1_dir
        self.edge_traces_dir = edge_traces_dir

    
    def extract_labels(self,  num_threads=8):
        """Runs concurrently using multiple threads."""

        programs = os.listdir(self.edge_traces_dir)
        for program in programs:
            df = self.process_program(program)
            self.save_df(df, program)


    def process_program(self, program_name):
        """"""
        njr_df = self.load_njr_file(program_name)
        mapping_data = self.load_segment_file(program_name)


         # Convert mapping JSON to a DataFrame if it's a list of dictionaries
        mapping_df = pd.DataFrame(mapping_data)

         # Drop duplicates in mapping based on 'edge_name'
        mapping_df = mapping_df.drop_duplicates(subset=['edge_name'], keep="first")

        # Drop duplicates in njr_df based on 'method' (src) and 'target'
        njr_df = njr_df.drop_duplicates(subset=['method', 'target'], keep='first')

        # Apply reformatting to src and target in mapping_df
        mapping_df["src"] = mapping_df["src"].apply(self.reformat_node_string)
        mapping_df["target"] = mapping_df["target"].apply(self.reformat_node_string)

        # Select only the required columns from njr_df before merging
        njr_df = njr_df[["method", "target", "wala-cge-0cfa-noreflect-intf-direct", "wiretap"]]

        # Rename 'method' in njr_df to 'src' for merging
        njr_df = njr_df.rename(columns={"method": "src"})
        njr_df = njr_df.rename(columns={"wala-cge-0cfa-noreflect-intf-direct": "direct"})

        # Merge NJR data with mapping data on 'src' and 'target'
        merged_df = njr_df.merge(mapping_df, on=["src", "target"], how="left")

        # Add trace column: 1 if there is a matching entry in mapping_df (i.e., 'edge_name' exists), else 0
        merged_df["trace"] = merged_df["edge_name"].notna().astype(int)

        # Fill missing values for 'direct' and 'wiretap' (default to 0)
        merged_df[["direct", "wiretap"]] = merged_df[["direct", "wiretap"]].fillna(0).astype(int)

        # Add program_name column
        merged_df["program_name"] = program_name

        # Remove rows where all three columns (direct, wiretap, trace) are 0
        merged_df = merged_df[~((merged_df["direct"] == 0) & (merged_df["wiretap"] == 0) & (merged_df["trace"] == 0))]

        # Select and reorder the required columns
        result_df = merged_df[[ "trace", "direct", "wiretap","edge_name", "src", "target", "program_name"]]

        return result_df
        

    
    def load_segment_file(self, program_name):
        '''loads features of each program along with its mapping json'''

        program_dir = os.path.join(self.edge_traces_dir, program_name)

        mapping_file = os.path.join(program_dir, "segments.json")

        
        if not os.path.exists(mapping_file):
            print(f"Missing mapping file in {program_dir}. Skipping.")
            return None
       
        
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
        except Exception as e:
            print(f"Error reading mapping file in {program_dir}: {e}")
            return None
        
        return mapping_data


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


    def save_df(self, df, program_name):
        '''saves df in a csv file'''

        path = os.path.join(self.edge_traces_dir, program_name, 'output.csv')
        df.to_csv(path, index=False)


if __name__ == '__main__':

    data_folder = '/home/mohammad/projects/CallGraphPruner_data'  
    NJR1_DATASET_FOLDER = f'{data_folder}/dataset-high-precision-callgraphs/full_callgraphs_set'

    edge_traces_dir = '/home/mohammad/projects/CallGraphPruner/data/edge-traces/cgs'

    # Create a DatasetGeneration instance and generate the dataset.
    le = LabelExtraction(njr1_dir=NJR1_DATASET_FOLDER, edge_traces_dir=edge_traces_dir)
    le.extract_labels()
