import os
import re
import json
import re

def build_index(input_file):
    """
    Build an index from the input file.
    """
    index = {}
    count = 0
    segments = []

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
                            "target": target
                        }
                        segments.append(data)

    # Remove instructions that do not have any endlines
    # index = {key: value for key, value in index.items() if value["endlines"]}

    with open('segments.json', 'w') as f:
        json.dump(segments, f)

    return segments



def safe_filename(s, max_length=50):
    """
    Sanitize a string so it can be safely used as part of a filename.
    Non-alphanumeric characters are replaced with underscores and the string is truncated.
    """
    safe = re.sub(r'[^\w\-\.]', '_', s)
    return safe[:max_length]

def extract_regions(input_file, segments, output_dir):
    """
    Extract integer pair lines from input_file for each region.
    
    Each region is written to its own file (named using the segment and region IDs).
    Only lines that do not start with "invoke:" or "addEdge:" are written.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open an output file for each region.
    # file_handles = {}
    buffers = {}
    for edge in segments:
        # filename = f"{edge['edge_name']}.txt"
        # out_path = os.path.join(output_dir, filename)
        # file_handles[edge['edge_name']] = open(out_path, 'w')
        buffers[edge['edge_name']] = []

    # # For efficiency, sort regions by start line.
    sorted_data = sorted(segments, key=lambda item: item["startline"])
    # Get the first region's start to know when to start processing.
    first_startline = sorted_data[0]['startline']

    with open(input_file, 'r') as f:
        for line_no, line in enumerate(f, start=1):
            # Skip lines before the first invoke.
            if first_startline is None or line_no < first_startline:
                continue

            # Only process integer pair lines (ignore special commands).
            if line.startswith("AgentLogger|visitinvoke:") or line.startswith("AgentLogger|addEdge:"):
                continue

            # For each region, check if the current line falls within its boundaries.
            # (Since regions likely do not overlap, this simple check is acceptable.)
            for inx in segments:
                if inx["startline"] <= line_no <= inx["endline"]:
                    # file_handles[inx['edge_name']].write(line)
                    buffers[inx['edge_name']].append(line)
                # elif inx['endline'] < line_no:
                    # file_handles[inx['edge_name']].close()
                    # filename = f"{inx['edge_name']}.log"
                    # out_path = os.path.join(output_dir, filename)

                    # with open(out_path, 'w') as fin:
                    #     fin.write("".join(buffers[inx['edge_name']]))

                    # buffers[inx['edge_name']].clear()

    for inx in segments:
        filename = f"{inx['edge_name']}.log"
        out_path = os.path.join(output_dir, filename)

        with open(out_path, 'w') as fin:
            fin.write("".join(buffers[inx['edge_name']]))
        
    # Close all open files.
    # for fh in file_handles.values():
    #     fh.close()

def main():
    input_file = "/home/mohammad/projects/CallGraphPruner/data/encoded/url72c32f3c54_Quickhull3d_quickhull3d_tgz-pJ8-com_github_quickhull3d_SimpleExampleJ8.txt.encoded"     # Replace with your actual input file path.
    output_dir = "regions_output"    # Directory where extracted regions will be stored.
    
    # First pass: Build the index.
    index = build_index(input_file)
    print("Index built:")
    
    # Second pass: Extract the regions to separate files.
    # extract_regions(input_file, index, output_dir)
    # print("Extraction complete.")

if __name__ == "__main__":
    main()
