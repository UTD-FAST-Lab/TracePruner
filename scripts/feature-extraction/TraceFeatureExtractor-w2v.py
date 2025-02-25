import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from gensim.models import Word2Vec

class TraceFeatureExtractor:
    def __init__(self, encoded_edge_traces_dir, vector_size=64, window=5, min_count=1, workers=4):
        self.encoded_edge_traces_dir = encoded_edge_traces_dir
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None  # Word2Vec model

    def load_trace(self, edge_trace_filepath):
        """Loads trace from file and converts it into a sequence of (int, int) pairs."""
        with open(edge_trace_filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # For each valid line, convert the (int, int) pair into a string "int_int"
        return [
            f"{x[0]}_{x[1]}"
            for line in lines
            if len(line.split(',')) == 2 and all(part.strip().isdigit() for part in line.split(','))
            for x in [tuple(map(int, line.split(',')))]
        ]

    def load_all_traces(self, max_programs=2):
        """
        Loads traces for training Word2Vec.
        Only loads traces from the first `max_programs` programs.
        """
        all_sequences = []
        programs = os.listdir(self.encoded_edge_traces_dir)[:max_programs]  # Limit to first two programs
        for program in programs:
            program_dir = os.path.join(self.encoded_edge_traces_dir, program, 'edges')
            if not os.path.isdir(program_dir):
                continue
            for filename in os.listdir(program_dir):
                file_path = os.path.join(program_dir, filename)
                trace = self.load_trace(file_path)
                if trace:  # Only add non-empty traces
                    all_sequences.append(trace)
        return all_sequences

    def train_word2vec(self):
        """Trains a Word2Vec model using a subset of traces."""
        print("🧠 Training Word2Vec model on a subset of traces...")
        all_sequences = self.load_all_traces(max_programs=2)
        self.model = Word2Vec(
            sentences=all_sequences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        print(f"✅ Word2Vec model trained on {len(all_sequences)} traces.")

    def encode_trace(self, trace):
        """Encodes a trace into a fixed-size feature vector using Word2Vec."""
        vectors = [self.model.wv[word] for word in trace if word in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)

    def process_program(self, program):
        """Processes a single program: loads traces, extracts features, and saves results."""
        program_dir = os.path.join(self.encoded_edge_traces_dir, program)
        edges_dir = os.path.join(program_dir, 'edges')
        if not os.path.isdir(edges_dir):
            return  # Skip if edges directory does not exist

        all_program_deep_features = []
        for edge_trace_filename in os.listdir(edges_dir):
            edge_trace_filepath = os.path.join(edges_dir, edge_trace_filename)
            trace = self.load_trace(edge_trace_filepath)
            feature_vector = self.encode_trace(trace)
            all_program_deep_features.append((edge_trace_filename, feature_vector))

        self.save_program_features(program_dir, all_program_deep_features)

    def save_program_features(self, output_dir, all_program_deep_features):
        """Saves extracted feature vectors to a CSV file."""
        output_file = os.path.join(output_dir, 'w2v_features.csv')
        with open(output_file, 'w', encoding='utf-8') as f:
            header = "edge_name," + ",".join([f"Feature_{i}" for i in range(self.vector_size)]) + "\n"
            f.write(header)
            for edge_name, vector in all_program_deep_features:
                f.write(f"{edge_name},{','.join(map(str, vector))}\n")

    def process_all_programs(self, num_threads=8):
        """Runs all programs concurrently using multiple threads."""
        if not self.model:
            self.train_word2vec()
        programs = os.listdir(self.encoded_edge_traces_dir)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(self.process_program, programs)

if __name__ == "__main__":
    encoded_edge_traces_dir = '/home/mohammad/projects/CallGraphPruner/data/encoded-edge'
    extractor = TraceFeatureExtractor(encoded_edge_traces_dir)
    extractor.process_all_programs()
    print("✅ Feature extraction complete!")
