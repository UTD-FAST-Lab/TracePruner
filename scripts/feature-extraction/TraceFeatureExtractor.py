import os, json, re, argparse, tempfile
import numpy as np
import networkx as nx
from collections import Counter
from sklearn.preprocessing import StandardScaler

class TraceFeatureExtractor:
    def __init__(self, encoded_edge_traces_dir):
        self.encoded_edge_traces_dir = encoded_edge_traces_dir
    
    def load_trace(self, edge_trace_filepath):
        """Loads trace from file and converts it into lists of (int, int) pairs."""
        if edge_trace_filepath.endswith(".log"):
            with open(edge_trace_filepath, 'r') as f:
                lines = [tuple(map(int, line.strip().split(','))) for line in f]
        return lines
    
    def extract_features(self, trace):
        """Extracts features from trace and stores them as feature vectors."""
        
        graph = self.build_graph(trace)
        feature_vector = self.compute_features(graph, trace)
        # self.feature_vectors.append((filename, feature_vector))
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
    
    def save_program_features(self, output_dir, all_program_features):
        """Saves extracted feature vectors to a CSV file."""
        output_file = os.path.join(output_dir, 'features.csv')

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("edge_name,num_nodes,num_edges,avg_degree,density,avg_pagerank,mc1,mc2,mc3,mc4,mc5,unique_funcs,entropy\n")
            for edge_name, vector in all_program_features:
                f.write(f"{edge_name},{','.join(map(str, vector))}\n")

    def process_programs(self):
        """
        Process each program in the encoded_edge_traces directory.
        - For each program, find the edges folder.
        - Extract features from each edge trace file.
        - Save the features to 'features.csv' in the program's folder.
        """
        for program in os.listdir(self.encoded_edge_traces_dir):
            program_dir = os.path.join(self.encoded_edge_traces_dir, program)
            edges_dir = os.path.join(program_dir, 'edges')

            if not os.path.isdir(edges_dir):
                continue  # Skip if edges_dir does not exist

            all_program_features = []

            for edge_trace_filename in os.listdir(edges_dir):
                edge_trace_filepath = os.path.join(edges_dir, edge_trace_filename)
                trace = self.load_trace(edge_trace_filepath)
                edge_features = self.extract_features(trace)

                all_program_features.append((edge_trace_filename, edge_features))

            self.save_program_features(program_dir, all_program_features)

class TraceEncoder:
    '''extracts the the traces of each edge and encode the names of the functions using a map'''


    def __init__(self, HASH_MAP_FILE, traces_dir, encoded_traces_dir, encoded_edge_traces_dir):
        self.HASH_MAP_FILE = HASH_MAP_FILE

        # create the folder structure for actual edge traces
        # base_dir = os.path.join(encoded_traces_dir, program_name.split('.')[0])
        # edges_dir = os.path.join(base_dir, "edges")
        # os.makedirs(edges_dir, exist_ok=True)

        # create the folder structure for encoded edge traces
        # encoded_base_dir = os.path.join(encoded_edge_traces_dir, program_name.split('.')[0])
        # encoded_edges_dir = os.path.join(encoded_base_dir, "edges")
        # os.makedirs(encoded_edges_dir, exist_ok=True)

        self.traces_dir = traces_dir
        self.encoded_traces_dir = encoded_traces_dir
        self.encoded_edge_traces_dir = encoded_edge_traces_dir


    def update_scg_mapping(self, edge_filepath, src, instruction, target):
        '''adds the edge to the mapping file'''
        
        edge = {
            'src': src,
            'instruction': instruction,
            'target': target,
            'edge_path': edge_filepath
        }

        return edge
    

    def extract_edge_trace(self, filename, encoded_edges_dir):
        '''extract the edges of the trace'''
        count = 0
        static_edges = []
        # Instead of storing invocation trace lines in a list,
        # we maintain a mapping of instruction -> temporary file (opened in text mode)
        current_invokes = {}
        # (The original invoke_traces dict wasn’t used, so we omit it.)

        file_path = os.path.join(self.encoded_traces_dir, filename)
        with open(file_path, 'r') as f:
            for line in f:
                # mimic the original strip() call (removes leading/trailing whitespace)
                line = line.strip()

                # --- Process "visitinvoke:" events ---
                if "AgentLogger|visitinvoke:" in line:
                    match = re.search(r'AgentLogger\|visitinvoke: (.+)', line)
                    if match:
                        instruction = match.group(1)
                        # (Re)initialize the invocation trace by using a temporary file.
                        # If an entry already exists for this instruction, close it and replace.
                        if instruction in current_invokes:
                            current_invokes[instruction].close()
                        # Open a new temporary file in text read/write mode.
                        # (It exists only during processing.)
                        temp_file = tempfile.TemporaryFile(mode='w+t')
                        current_invokes[instruction] = temp_file

                # --- Process "addEdge:" events ---
                if "AgentLogger|addEdge:" in line:
                    match = re.search(
                        r'addEdge: (Node: < .*? > Context: Everywhere) (.*?) (Node: < .*? > Context: Everywhere)',
                        line
                    )
                    if match:
                        src = match.group(1)
                        instruction = match.group(2)
                        target = match.group(3)

                        if instruction in current_invokes:
                            temp_file = current_invokes[instruction]
                            # Record the current file position (i.e. the end of the current trace)
                            pos_before = temp_file.tell()
                            # Temporarily write the addEdge line to the trace
                            temp_file.write(line + "\n")
                            temp_file.flush()
                            pos_after = temp_file.tell()
                            # Read the full content (which now includes the addEdge line)
                            temp_file.seek(0)
                            trace_content = temp_file.read()
                            # Write the trace snapshot to a new edge file
                            count += 1
                            edge_filename = f"{count}.log"
                            edge_filepath = os.path.join(encoded_edges_dir, edge_filename)
                            with open(edge_filepath, 'w') as edge_file:
                                edge_file.write(trace_content + "\n\n")
                            # Update the mapping (edge file, src, instruction, target)
                            edge_entry = self.update_scg_mapping(edge_filepath, src, instruction, target)
                            static_edges.append(edge_entry)
                            # "Pop" the addEdge line by truncating the temporary file back to pos_before
                            temp_file.truncate(pos_before)
                            # Move the file pointer back to the end so further writes continue properly
                            temp_file.seek(0, os.SEEK_END)

                # --- Append the current line to every active invocation trace ---
                for instr, temp_file in current_invokes.items():
                    temp_file.write(line + "\n")
        # End reading the input trace

        # Close all temporary files (they’re no longer needed)
        for temp_file in current_invokes.values():
            temp_file.close()

        # Save the collected edge mappings to the mapping file
        map_dir = os.path.join(self.encoded_edge_traces_dir, filename.split('.')[0])
        os.makedirs(map_dir, exist_ok=True)
        map_path = os.path.join(map_dir, 'edge_map.json')
        with open(map_path, "w", encoding="utf-8") as map_file:
            json.dump(static_edges, map_file, indent=4)


    # def extract_edge_trace(self, filename, encoded_edges_dir):
    #     '''extract the edges of the trace'''
    #     count = 0   
    #     invoke_traces = {}
    #     current_invokes = {}
    #     static_edges = []
    #     file_path = os.path.join(self.encoded_traces_dir, filename)

    #     with open(file_path, 'r') as f:
    #         for line in f:
    #             line = line.strip()
                
    #             if "AgentLogger|visitinvoke:" in line:
    #                 match = re.search(r'AgentLogger\|visitinvoke: (.+)', line)
    #                 if match:
    #                     instruction = match.group(1)
    #                     current_invokes[instruction] = []
    #                     invoke_traces[instruction] = []
                    
    #             if "AgentLogger|addEdge:" in line:
    #                 match = re.search(r'addEdge: (Node: < .*? > Context: Everywhere) (.*?) (Node: < .*? > Context: Everywhere)', line)
    #                 if match:
    #                     src = match.group(1)  # Captures the source node
    #                     instruction = match.group(2)  # Captures the instruction
    #                     target = match.group(3)  # Captures the target node
                        
    #                     if instruction in current_invokes:
    #                         current_invokes[instruction].append(line)
                            
    #                         count += 1

    #                         edge_filename = f"{count}.log"
    #                         edge_filepath = os.path.join(encoded_edges_dir, edge_filename)
    #                         with open(edge_filepath, 'w') as f:
    #                             f.write("\n".join(current_invokes[instruction]) + "\n\n")

    #                         edge = self.update_scg_mapping(edge_filepath, src, instruction, target)
    #                         static_edges.append(edge)

    #                         current_invokes[instruction].pop()
                            
    #                         # current_invokes[instruction] = []  # Reset for the next edge
                
    #             for inst in current_invokes:
    #                 current_invokes[inst].append(line)

    #     # save the edges to the mapping file
    #     map_path = os.path.join(self.encoded_edge_traces_dir, filename.split('.')[0], 'edge_map.json')
    #     with open(map_path, "w", encoding="utf-8") as fin:
    #         json.dump(static_edges, fin, indent=4)

    def extract_edge_traces(self):

        for filename in os.listdir(self.encoded_traces_dir):
            if filename.endswith(".encoded"):

                # create the folder structure for encoded edge traces of this program
                encoded_base_dir = os.path.join(self.encoded_edge_traces_dir, filename.split('.')[0])
                encoded_edges_dir = os.path.join(encoded_base_dir, "edges")
                os.makedirs(encoded_edges_dir, exist_ok=True)

                self.extract_edge_trace(filename, encoded_edges_dir)


    def load_hash_map(self):
        """Load existing hash map from file, or create a new one."""
        if os.path.exists(self.HASH_MAP_FILE):
            with open(self.HASH_MAP_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["string_to_id"], {int(k): v for k, v in data["id_to_string"].items()}
        return {}, {}

    def save_hash_map(self, string_to_id, id_to_string):
        """Save hash map to a JSON file."""
        with open(self.HASH_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump({"string_to_id": string_to_id, "id_to_string": id_to_string}, f, indent=4)

    def encode_file(self, file_path, filename, string_to_id, id_to_string):
        """Encode a file, updating the hash map if needed."""
        unique_strings = set(string_to_id.keys())  # Get current unique strings

        encoded_path = os.path.join(self.encoded_traces_dir, filename)
        encoded_file = encoded_path + ".encoded"
        
        # Read file and process edges
        with open(file_path, "r", encoding="utf-8") as fin, open(encoded_file, "w", encoding="utf-8") as fw:
            for line in fin:
                line = line.strip()
                if line.startswith("AgentLogger|visitinvoke: ") or line.startswith("AgentLogger|addEdge: "):
                    fw.write(f'{line}\n')

                elif line.startswith("AgentLogger|CG_edge: "):
                    # Remove the prefix before splitting
                    line = line[len("AgentLogger|CG_edge: "):]

                    # Split by "->" with optional spaces
                    parts = re.split(r"\s*->\s*", line)
                    if len(parts) == 2:
                        left, right = parts
                        
                        # Assign new IDs if needed
                        for s in [left, right]:
                            if s not in unique_strings:
                                new_id = len(string_to_id) + 1
                                string_to_id[s] = new_id
                                id_to_string[new_id] = s
                                unique_strings.add(s)

                        # Save encoded results
                        fw.write(f"{string_to_id[left]},{string_to_id[right]}\n")

    
    def decode_edges(self, encoded_edges, id_to_string):
        """Decode numeric edges back into text format."""
        return [(id_to_string[left], id_to_string[right]) for left, right in encoded_edges]


    def process_files(self):
        """Process multiple files, encoding each one while maintaining a consistent hash map."""
        string_to_id, id_to_string = self.load_hash_map()
        
        for filename in os.listdir(self.traces_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.traces_dir, filename)
                self.encode_file(file_path, filename, string_to_id, id_to_string)

        # Save updated hash map for future use
        self.save_hash_map(string_to_id, id_to_string)


def parse_cmd_arguments():
    '''parses the command lines arguments'''

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process some command-line arguments.")

    # Add arguments
    # parser.add_argument("--program", type=str, help="name of the target program")
    parser.add_argument("--trace", action="store_true", help="Enable trace extraction")
    parser.add_argument("--feature", action="store_true", help="Enable feature extraction")

    # Parse arguments
    args = parser.parse_args()

    return args

    
if __name__ == "__main__":

    # parse arguments
    args = parse_cmd_arguments()

    encoded_edge_traces_dir = '/home/mohammad/projects/CallGraphPruner/data/encoded-edge'

    if args.trace:
        # extracting traces of indivudaul edges and encoding them.
        wala_hash_map_path = '/home/mohammad/projects/CallGraphPruner/scripts/WALA_hash_map.json'
        traces_dir = '/home/mohammad/projects/CallGraphPruner/data/cgs' 
        encoded_traces_dir = '/home/mohammad/projects/CallGraphPruner/data/encoded'           

        tc = TraceEncoder(wala_hash_map_path, traces_dir, encoded_traces_dir, encoded_edge_traces_dir)
        # tc.process_files()
        tc.extract_edge_traces()


    if args.feature:
        # extracting features from the traces
        encoded_edge_traces_dir = f'/home/mohammad/projects/CallGraphPruner/data/encoded-edge/'    
        
        extractor = TraceFeatureExtractor(encoded_edge_traces_dir)
        extractor.process_programs()
