import os, json, re, argparse, tempfile
import hashlib
import numpy as np
import networkx as nx
from collections import Counter

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class TraceFeatureExtractor:

    def __init__(self, encoded_edge_traces_dir):
        self.encoded_edge_traces_dir = encoded_edge_traces_dir

        self.model = Sequential([
            Input(shape=(None, 2)),  # Explicitly define input shape
            LSTM(32, return_sequences=False),
            Dense(16, activation='relu'),
            Dense(8, activation='relu')
        ])

    
    def load_trace(self, edge_trace_filepath):
        """Loads trace from file and converts it into lists of (int, int) pairs."""
        lines = []  # Store valid (int, int) pairs

        with open(edge_trace_filepath, 'r') as f:
            lines = f.readlines()  # Read all lines at once

        # Process lines: Strip, filter invalid entries, and convert to (int, int)
        return [
            tuple(map(int, line.strip().split(',')))
            for line in lines
            if line.strip() and line.lstrip()[0].isdigit() and len(line.split(',')) == 2
        ]

    
    def extract_features(self, trace):
        """Extracts features from trace and stores them as feature vectors."""

        lstm_features = self.compute_lstm_features(trace)
        
        graph = self.build_graph(trace)
        feature_vector = self.compute_features(graph, trace)
        return feature_vector, lstm_features
    
    def compute_lstm_features(self, sequence):

        X = np.array(sequence).reshape(1, len(sequence), 2)

        # Get feature vector
        feature_vector = self.model.predict(X).flatten().tolist()  # Flatten to 1D

        return feature_vector

        
    def build_graph(self, trace):
        """Constructs a directed graph from the trace data."""
        G = nx.DiGraph()
        for src, dst in trace:
            G.add_edge(src, dst)
        return G
    
    def compute_features(self, G, trace):
        """Computes graph-based, sequence-based, and statistical features."""
        # Graph features
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        avg_degree = np.mean([deg for _, deg in G.degree()]) if num_nodes > 0 else 0
        density = nx.density(G)
        pagerank_scores = nx.pagerank(G) if num_nodes > 0 else {}
        avg_pagerank = np.mean(list(pagerank_scores.values())) if pagerank_scores else 0
        
        # Sequence-based features
        transitions = Counter(trace)
        most_common_trans = transitions.most_common(5)
        most_common_counts = [count for _, count in most_common_trans] + [0] * (5 - len(most_common_trans))
        
        # Statistical features
        unique_functions = len(set([func for pair in trace for func in pair]))
        entropy = -sum((count / len(trace)) * np.log2(count / len(trace)) for count in transitions.values())
        
        return [num_nodes, num_edges, avg_degree, density, avg_pagerank] + most_common_counts + [unique_functions, entropy]
    
    def save_program_features(self, output_dir, all_program_features, all_program_lstm_features):
        """Saves extracted feature vectors to a CSV file."""
        output_file = os.path.join(output_dir, 'features.csv')
        output_lstm_file = os.path.join(output_dir, 'lstm_features.csv')
        output_combined_file = os.path.join(output_dir, 'combined.csv')

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("edge_name,num_nodes,num_edges,avg_degree,density,avg_pagerank,mc1,mc2,mc3,mc4,mc5,unique_funcs,entropy\n")
            for edge_name, vector in all_program_features:
                f.write(f"{edge_name},{','.join(map(str, vector))}\n")

        with open(output_lstm_file, 'w', encoding='utf-8') as f:

            f.write("edge_name," + ",".join([f"Feature_{i}" for i in range(8)]) + "\n")

            for edge_name, vector in all_program_lstm_features:
                f.write(f"{edge_name},{','.join(map(str, vector))}\n")



        combined_features = {}

        # Store the first set of features
        for edge_name, vector in all_program_features:
            combined_features[edge_name] = list(vector)  # Store as list for easy appending

        # Add LSTM-based features
        for edge_name, lstm_vector in all_program_lstm_features:
            if edge_name in combined_features:
                combined_features[edge_name].extend(lstm_vector)  # Append LSTM features
            else:
                # If an edge is in LSTM features but not in the first set, add with NaN for missing features
                combined_features[edge_name] = [None] * len(all_program_features[0][1]) + list(lstm_vector)

        # Write combined features to a CSV file
        with open(output_combined_file, 'w', encoding='utf-8') as f:
            # Create a header row
            num_basic_features = len(all_program_features[0][1])  # Get the number of basic features
            num_lstm_features = len(all_program_lstm_features[0][1])  # Get the number of LSTM features
            
            header = ["edge_name"] + [f"Feature_{i}" for i in range(num_basic_features)] + [f"LSTM_Feature_{i}" for i in range(num_lstm_features)]
            f.write(",".join(header) + "\n")

            # Write each row
            for edge_name, feature_vector in combined_features.items():
                f.write(f"{edge_name},{','.join(map(str, feature_vector))}\n")

        print(f"Combined features saved to {output_combined_file}")

    # def process_programs(self):
    #     """
    #     Process each program in the encoded_edge_traces directory.
    #     - For each program, find the edges folder.
    #     - Extract features from each edge trace file.
    #     - Save the features to 'features.csv' in the program's folder.
    #     """
    #     for program in os.listdir(self.encoded_edge_traces_dir):
    #         program_dir = os.path.join(self.encoded_edge_traces_dir, program)
    #         edges_dir = os.path.join(program_dir, 'edges')

    #         print(program)

    #         if not os.path.isdir(edges_dir):
    #             continue  # Skip if edges_dir does not exist

    #         excluded_edges = self.eliminate_unmatched_edges(program_dir)

    #         all_program_features = []

    #         for edge_trace_filename in os.listdir(edges_dir):
    #             if edge_trace_filename not in excluded_edges:
    #                 print(edge_trace_filename)
    #                 edge_trace_filepath = os.path.join(edges_dir, edge_trace_filename)
    #                 trace = self.load_trace(edge_trace_filepath)
    #                 edge_features = self.extract_features(trace)

    #                 all_program_features.append((edge_trace_filename, edge_features))

    #         self.save_program_features(program_dir, all_program_features)
    #         print(program, ': done')

    def process_program(self, program):
        """Processes a single program: loads traces, extracts features, and saves results."""
        program_dir = os.path.join(self.encoded_edge_traces_dir, program)
        edges_dir = os.path.join(program_dir, 'edges')

        if not os.path.isdir(edges_dir):
            return  # Skip if edges_dir does not exist

        all_program_features = []
        all_program_lstm_features = []

        for edge_trace_filename in os.listdir(edges_dir):

            edge_trace_filepath = os.path.join(edges_dir, edge_trace_filename)
            trace = self.load_trace(edge_trace_filepath)
            edge_features, lstm_features = self.extract_features(trace)
            all_program_features.append((edge_trace_filename, edge_features))
            all_program_lstm_features.append((edge_trace_filename, lstm_features))

        self.save_program_features(program_dir, all_program_features, all_program_lstm_features)

    
    def process_all_programs(self, num_threads=8):
        """Runs all programs concurrently using multiple threads."""
        programs = os.listdir(self.encoded_edge_traces_dir)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(self.process_program, programs)


    
if __name__ == "__main__":


    # encoded_edge_traces_dir = '/home/mohammad/projects/CallGraphPruner/data/encoded-edge'
    
    # extracting features from the edge traces
    encoded_edge_traces_dir = f'/home/mohammad/projects/CallGraphPruner/data/encoded-edge/'    
    extractor = TraceFeatureExtractor(encoded_edge_traces_dir)
    extractor.process_all_programs()
