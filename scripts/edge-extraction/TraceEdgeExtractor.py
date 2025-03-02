import os, json, re

import pandas as pd


class TraceEdgeExtractor:
    '''extracts the the traces of each edge and encode the names of the functions using a map'''


    def __init__(self, HASH_MAP_FILE, traces_dir, encoded_traces_dir, encoded_edge_traces_dir, trace_type):
        
        self.HASH_MAP_FILE = HASH_MAP_FILE
        self.traces_dir = traces_dir
        self.encoded_traces_dir = encoded_traces_dir
        self.encoded_edge_traces_dir = encoded_edge_traces_dir
        self.trace_type = trace_type


    def build_index(self, filename):
        """
        Build an index from the input file.
        """
        index = {}
        count = 0
        segments = []


        input_file = os.path.join(encoded_traces_dir, filename)
        with open(input_file, 'r') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if line.startswith("AgentLogger|visitinvoke:"):
                    match = re.search(r'AgentLogger\|visitinvoke: (.+)', line)
                    if match:
                        instruction = match.group(1)
                        index[instruction] =  line_no + 1
                            

                elif line.startswith("AgentLogger|addEdge:"):
                    match = re.search(
                        r'addEdge: (Node: < .*? > Context: Everywhere) (.*?) (Node: < .*? > Context: Everywhere)',
                        line
                    )
                    if match:
                        src = match.group(1)
                        instruction = match.group(2)
                        target = match.group(3)

                        if instruction in index.keys():
                            count += 1
                            data = {
                                "startline": index[instruction],
                                "endline": line_no - 1,
                                "edge_name": count,
                                "src": src,
                                "target": target,
                                "instruction": instruction
                            }
                            
                            segments.append(data)


        return self.process_segments(segments, filename)

    
    def process_segments(self, segments, filename):
        '''eliminate unmatched edges + remove other edges' traces from an edge'''

        filename = filename.split('.')[0]

        segments = self.remove_other_traces(segments)
        segments = self.eliminate_unmatched_edges(segments, filename)
        
        program_dir = os.path.join(self.encoded_edge_traces_dir, filename)
        segment_outputfile = os.path.join(program_dir, 'segments.json')
        with open(segment_outputfile, 'w') as f:
            json.dump(segments, f)

        return segments

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
    

    def load_njr_file(self, program_name):
        '''load njr file that has the labels'''

        program_dir = os.path.join('/home/mohammad/projects/CallGraphPruner_data/dataset-high-precision-callgraphs/full_callgraphs_set', program_name)
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


    def eliminate_unmatched_edges(self, edges, filename):
        '''eliminate the edges that are not matched with cgPruner's dataset'''

        njr_df =  self.load_njr_file(filename)

        edges = [edge for edge in edges if not njr_df[
            (njr_df['method'] == self.reformat_node_string(edge['src'])) & 
            (njr_df['target'] == self.reformat_node_string(edge['target']))
        ].empty]

        return edges
       

    def remove_other_traces(self, segments):
        '''
        For each segment, check if other segments have startlines and endlines inside this segment and remove them.
        The approach is:
        1. Find the nearest startline of another segment that is inside the current segment.
        2. Find the farthest endline of another segment inside the current segment.
        3. Keep only two parts:
        - From `current_startline` to `nearest_start - 1`
        - From `farthest_end + 1` to `current_endline`
        '''

        filtered_segments = []

        # Sort segments by startline for better processing
        segments.sort(key=lambda x: x["startline"])

        for i, seg in enumerate(segments):
            cur_start = seg["startline"]
            cur_end = seg["endline"]

            # Find the nearest start and the farthest end of any nested segment
            nearest_start = None
            farthest_end = None

            for j, other_seg in enumerate(segments):
                if i == j:
                    continue  # Skip itself

                other_start = other_seg["startline"]
                other_end = other_seg["endline"]

                if cur_start < other_start < cur_end:
                    if nearest_start is None or other_start < nearest_start:
                        nearest_start = other_start

                if cur_start < other_end < cur_end:
                    if farthest_end is None or other_end > farthest_end:
                        farthest_end = other_end

            # If we found nested segments, split only into two major parts
            if nearest_start and farthest_end:
                part1 = seg.copy()
                part1["endline"] = nearest_start - 1  # Before first nested segment

                part2 = seg.copy()
                part2["startline"] = farthest_end + 1  # After last nested segment

                filtered_segments.append(part1)
                filtered_segments.append(part2)
            else:
                filtered_segments.append(seg)

        return filtered_segments



    def extract_edges(self, segments, filename, output_dir):
        """Read input file once, store in buffer, and slice for each segment."""
        """Read input file once, store in buffer, and slice for each segment.
       Concatenates segments with the same edge_name into a single output file.
    """
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        # Read the entire file into a buffer (list of lines)
        input_file = os.path.join(self.encoded_traces_dir, filename)
        with open(input_file, 'r') as f:
            buffer = f.readlines()

        # Dictionary to store concatenated slices for each edge_name
        edge_slices = {}

        for segment in segments:
            start, end, edge_name = segment["startline"], segment["endline"], segment["edge_name"]

            # Extract the relevant slice from the buffer
            lines = buffer[start - 1 : end]  # Convert 1-based index to 0-based
            
            # Append the extracted lines to the corresponding edge_name key
            if edge_name not in edge_slices:
                edge_slices[edge_name] = []
            edge_slices[edge_name].extend(lines)

        # Write the aggregated slices to files
        for edge_name, lines in edge_slices.items():
            output_path = os.path.join(output_dir, f"{edge_name}.log")
            with open(output_path, 'w') as out_file:
                out_file.writelines(lines)
        


    def extract_edge_traces(self):

        for filename in os.listdir(self.encoded_traces_dir):
            if filename.endswith(".encoded"):

                # create the folder structure for encoded edge traces of this program
                encoded_base_dir = os.path.join(self.encoded_edge_traces_dir, filename.split('.')[0])
                encoded_edges_dir = os.path.join(encoded_base_dir, "edges")
                os.makedirs(encoded_edges_dir, exist_ok=True)

                segments = self.build_index(filename)
                self.extract_edges(segments, filename, encoded_edges_dir)


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
                    continue

                if self.trace_type == 'cgs':

                    if line.startswith("AgentLogger|CG_edge: "):
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

                elif self.trace_type == 'branches':
       
                    if line.startswith("AgentLogger|BRANCH: "):
                        # Remove the prefix before matching
                        line = line[len("AgentLogger|BRANCH: "):]

                        # Use a regex to capture the method signature and the IF branch ID
                        m = re.match(r"^(.*):IF#(\d+)$", line)
                        if m:
                            method, branch_id = m.group(1), m.group(2)
                            
                            # Assign new IDs if needed
                            if method not in unique_strings:
                                new_id = len(string_to_id) + 1
                                string_to_id[method] = new_id
                                id_to_string[new_id] = method
                                unique_strings.add(method)

                            # Save encoded results
                            fw.write(f"{string_to_id[method]},{branch_id}\n")



    
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



if __name__ == "__main__":

    trace_types = ('cgs', 'branches', 'variables')
    trace_type = trace_types[1]

    wala_hash_map_path = '/home/mohammad/projects/CallGraphPruner/scripts/WALA_hash_map.json'


    traces_dir = f'/home/mohammad/projects/CallGraphPruner/data/traces/{trace_type}' 
    encoded_traces_dir = f'/home/mohammad/projects/CallGraphPruner/data/encoded/{trace_type}'           
    encoded_edge_traces_dir = f'/home/mohammad/projects/CallGraphPruner/data/encoded-edge/{trace_type}'  

    tc = TraceEdgeExtractor(wala_hash_map_path, traces_dir, encoded_traces_dir, encoded_edge_traces_dir, trace_type)

    # encode the trace
    tc.process_files()

    # extract the edge traces 
    tc.extract_edge_traces()