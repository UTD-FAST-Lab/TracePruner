import json
import os

def read_json(json_file):
    """Load JSON data from a file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def process_file(json_file, input_file, output_dir):
    """Read input file once, store in buffer, and slice for each segment."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Read the entire file into a buffer (list of lines)
    with open(input_file, 'r') as f:
        buffer = f.readlines()

    # Read JSON segments
    segments = read_json(json_file)

    for segment in segments:
        start, end, edge_name = segment["startline"], segment["endline"], segment["edge_name"]
        
        # Extract the relevant slice from the buffer
        lines = buffer[start - 1 : end]  # Convert 1-based index to 0-based
        
        # Define output file path
        output_path = os.path.join(output_dir, f"{edge_name}.log")
        
        # Write extracted lines to a file
        with open(output_path, 'w') as out_file:
            out_file.writelines(lines)

            
# Example usage
json_path = "segments.json"   # Path to your JSON file
input_path = "/home/mohammad/projects/CallGraphPruner/data/encoded/url72c32f3c54_Quickhull3d_quickhull3d_tgz-pJ8-com_github_quickhull3d_SimpleExampleJ8.txt.encoded"    # Path to the input file
output_directory = "output"   # Directory where sliced files will be saved

process_file(json_path, input_path, output_directory)
