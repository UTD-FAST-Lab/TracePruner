import os
import numpy as np
import networkx as nx
from collections import Counter
from sklearn.preprocessing import StandardScaler

class TraceFeatureExtractor:
    def __init__(self, trace_folder):
        self.trace_folder = trace_folder
        self.feature_vectors = []
    
    def load_traces(self):
        """Loads traces from files and converts them into lists of (int, int) pairs."""
        traces = {}
        for filename in os.listdir(self.trace_folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.trace_folder, filename)
                with open(filepath, 'r') as f:
                    lines = [tuple(map(int, line.strip().split(','))) for line in f]
                    traces[filename] = lines
        return traces
    
    def extract_features(self, traces):
        """Extracts features from traces and stores them as feature vectors."""
        for filename, trace in traces.items():
            graph = self.build_graph(trace)
            feature_vector = self.compute_features(graph, trace)
            self.feature_vectors.append((filename, feature_vector))
        
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
    
    def save_features(self, output_file):
        """Saves extracted feature vectors to a CSV file."""
        with open(output_file, 'w') as f:
            f.write("filename,num_nodes,num_edges,avg_degree,density,avg_pagerank,mc1,mc2,mc3,mc4,mc5,unique_funcs,entropy\n")
            for filename, vector in self.feature_vectors:
                f.write(f"{filename},{','.join(map(str, vector))}\n")
    
if __name__ == "__main__":
    trace_folder = "path/to/trace/files"
    output_file = "features.csv"
    
    extractor = TraceFeatureExtractor(trace_folder)
    traces = extractor.load_traces()
    extractor.extract_features(traces)
    extractor.save_features(output_file)
    print("Feature extraction complete! Features saved to", output_file)
