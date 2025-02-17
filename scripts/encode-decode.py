import re
import json
import os
import ast

HASH_MAP_FILE = "WALA_hash_map.json"
ORIGINAL_TRACE_DIR_PATH = '../data/cgs'
ENCODED_TRACE_PATH = '../data/encoded'

def load_hash_map():
    """Load existing hash map from file, or create a new one."""
    if os.path.exists(HASH_MAP_FILE):
        with open(HASH_MAP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["string_to_id"], {int(k): v for k, v in data["id_to_string"].items()}
    return {}, {}

def save_hash_map(string_to_id, id_to_string):
    """Save hash map to a JSON file."""
    with open(HASH_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump({"string_to_id": string_to_id, "id_to_string": id_to_string}, f, indent=4)

def encode_file(file_path, string_to_id, id_to_string):
    """Encode a file, updating the hash map if needed."""
    encoded_edges = []
    unique_strings = set(string_to_id.keys())  # Get current unique strings
    
    # Read file and process edges
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("AgentLogger|CG_edge: "):
                continue  # Skip lines that do not match

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

                # Encode edges
                encoded_edges.append(string_to_id[left], string_to_id[right])

    return encoded_edges

def decode_edges(encoded_edges, id_to_string):
    """Decode numeric edges back into text format."""
    return [(id_to_string[left], id_to_string[right]) for left, right in encoded_edges]

def process_files(file_paths):
    """Process multiple files, encoding each one while maintaining a consistent hash map."""
    string_to_id, id_to_string = load_hash_map()
    
    for file_path in file_paths:
        encoded_path = os.path.join(ENCODED_TRACE_PATH, file_path)
        file_path = os.path.join(ORIGINAL_TRACE_DIR_PATH, file_path)
        encoded_edges = encode_file(file_path, string_to_id, id_to_string)
        
        # Save encoded results
        encoded_file = encoded_path + ".encoded"
        with open(encoded_file, "w", encoding="utf-8") as f:
            for edge in encoded_edges:
                f.write(f"{edge}\n")
        
        print(f"Encoded {file_path} -> {encoded_file}")

    # Save updated hash map for future use
    save_hash_map(string_to_id, id_to_string)



# Example usage
file_list = ["VC1.txt","VC2.txt","VC3.txt","VC4.txt","MT1.txt","MT2.txt", "MT3.txt"]  # List your files here
process_files(file_list)  # Encode all files

