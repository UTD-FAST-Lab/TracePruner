import json
import os, json, re
import numpy as np
import pandas as pd

# program_dir = '/home/mohammad/projects/CallGraphPruner/data/encoded/url72c32f3c54_Quickhull3d_quickhull3d_tgz-pJ8-com_github_quickhull3d_Simp'
program_name = 'url72c32f3c54_Quickhull3d_quickhull3d_tgz-pJ8-com_github_quickhull3d_SimpleExampleJ8'

def build_index():
    """
    Build an index from the input file.
    """
    index = {}
    count = 0
    segments = []

    encoded_traces_dir = '/home/mohammad/projects/CallGraphPruner/data/encoded'

    filename = program_name + '.txt.encoded'
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

    # Remove instructions that do not have any endlines
    # index = {key: value for key, value in index.items() if value["endlines"]}

    process_segments(segments)

    # 

    # return segments

def reformat_node_string(node_string):
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
    

def load_njr_file(program_name):
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


def eliminate_unmatched_edges(edges):
    '''eliminate the edges that are not matched with cgPruner's dataset'''

    njr_df =  load_njr_file(program_name)

    edges = [edge for edge in edges if not njr_df[
        (njr_df['method'] == reformat_node_string(edge['src'])) & 
        (njr_df['target'] == reformat_node_string(edge['target']))
    ].empty]


    return edges


def remove_other_traces(segments):
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


def process_segments(segments):
    '''eliminate unmatched edges + remove other edges' traces from an edge'''

    segments = remove_other_traces(segments)
    segments = eliminate_unmatched_edges(segments)
    
    program_dir = '/home/mohammad/projects/CallGraphPruner/scripts/edge-extraction/output'
    segment_outputfile = os.path.join(program_dir, 'segments.json')
    with open(segment_outputfile, 'w') as f:
        json.dump(segments, f)



build_index()