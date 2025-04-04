import os, json, re

import pandas as pd
import argparse



class TraceEdgeExtractor:
    '''extracts the the traces of each edge and encode the names of the functions using a map'''


    def __init__(self, traces_dir, edge_traces_dir, trace_type):
        
        self.traces_dir = traces_dir
        self.edge_traces_dir = edge_traces_dir
        self.trace_type = trace_type


    def build_index(self, filename):
        """
        Build an index from the input file.
        """
        index = {}
        count = 0
        segments = []


        input_file = os.path.join(traces_dir, filename)
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
        # segments = self.eliminate_unmatched_edges(segments, filename)
        
        program_dir = os.path.join(self.edge_traces_dir, filename)
        segment_outputfile = os.path.join(program_dir, 'segments.json')
        with open(segment_outputfile, 'w') as f:
            json.dump(segments, f)

        return segments

    # def reformat_node_string(self, node_string):
    #     pattern = r"Node: < (?:Primordial|Application), ([^,]+), ([^ ]+) >"
    #     match = re.search(pattern, node_string)
        
    #     if match:
    #         class_name = match.group(1)
    #         method_signature = match.group(2)
            
    #         # Remove the leading 'L' from the class name if present
    #         class_name = class_name.lstrip('L')
            
    #         # Replace the space before method signature with ':'
    #         formatted_string = f"{class_name}.{method_signature.split('(')[0]}:({method_signature.split('(')[1]}"
    #         return formatted_string
    #     return None  # Return None if the format is incorrect
    

    # def load_njr_file(self, program_name):
    #     '''load njr file that has the labels'''

    #     program_dir = os.path.join('/home/mohammad/projects/CallGraphPruner_data/dataset-high-precision-callgraphs/full_callgraphs_set', program_name)
    #     njr1_file = os.path.join(program_dir, 'wala0cfa.csv')

    #     if not os.path.exists(njr1_file):
    #         print(f"Missing njr file in {program_dir}. Skipping.")
    #         return None
        
    #     # Load the file
    #     try:
    #         njr_df = pd.read_csv(njr1_file)
    #     except Exception as e:
    #         print(f"Error reading njr file in {program_dir}: {e}")
    #         return None

    #     return njr_df


    # def eliminate_unmatched_edges(self, edges, filename):
    #     '''eliminate the edges that are not matched with cgPruner's dataset'''

    #     njr_df =  self.load_njr_file(filename)

    #     edges = [edge for edge in edges if not njr_df[
    #         (njr_df['method'] == self.reformat_node_string(edge['src'])) & 
    #         (njr_df['target'] == self.reformat_node_string(edge['target']))
    #     ].empty]

    #     return edges
       

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
        """Read input file once, store in buffer, and slice for each segment.
            Concatenates segments with the same edge_name into a single output file.
        """
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        # Read the entire file into a buffer (list of lines)
        input_file = os.path.join(self.traces_dir, filename)
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

        for filename in os.listdir(self.traces_dir):
            if filename.endswith(".txt"):

                # create the folder structure for edge traces of this program
                base_dir = os.path.join(self.edge_traces_dir, filename.split('.')[0])
                edges_dir = os.path.join(base_dir, "edges")
                os.makedirs(edges_dir, exist_ok=True)

                segments = self.build_index(filename)
                self.extract_edges(segments, filename, edges_dir)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract edges for different instrumentation types of WALA")
    # Define command-line arguments
    parser.add_argument('--type', type=str, required=True, help="specify the type of instrumentation (B:Branch, C:Call graph)")
    args = parser.parse_args()  # Parse arguments

    trace_types = ('cgs', 'branches', 'variables')
    
    if args.type == 'B':
        trace_type = trace_types[1]
    elif args.type == 'C':
        trace_type = trace_types[0]

    traces_dir = f'/home/mohammad/projects/CallGraphPruner/data/traces/{trace_type}'  
    edge_traces_dir = f'/home/mohammad/projects/CallGraphPruner/data/edge-traces/{trace_type}'  

    tc = TraceEdgeExtractor(traces_dir, edge_traces_dir, trace_type) #choose trace type

    # extract the edge traces 
    tc.extract_edge_traces()