import json
import os
from itertools import islice

def read_json(json_file):
    """Load JSON data from a file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def extract_lines(input_file, start, end):
    """Extract lines from start to end using islice."""
    with open(input_file, 'r') as f:
        return list(islice(f, start - 1, end))

def process_file(json_file, input_file, output_dir):
    """Process each segment from JSON and write extracted lines to separate files."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    segments = read_json(json_file)

    for segment in segments:
        start, end, edge_name = segment["startline"], segment["endline"], segment["edge_name"]
        
        lines = extract_lines(input_file, start, end)
        
        # Define output file path
        output_path = os.path.join(output_dir, f"{edge_name}.log")
        
        # Write to file
        with open(output_path, 'w') as out_file:
            out_file.writelines(lines)

# Example usage
json_path = "segments.json"   # Path to your JSON file
input_path = "/home/mohammad/projects/CallGraphPruner/data/encoded/url72c32f3c54_Quickhull3d_quickhull3d_tgz-pJ8-com_github_quickhull3d_SimpleExampleJ8.txt.encoded"    # Path to the input file
output_directory = "output"   # Directory where sliced files will be saved

process_file(json_path, input_path, output_directory)
