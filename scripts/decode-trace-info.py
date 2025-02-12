import re
import json
import os
import ast

HASH_MAP_FILE = "WALA_hash_map.json"
ENCODED_TRACE_PATH = '../data/encoded'

def load_hash_map():
    """Load existing hash map from file, or create a new one."""
    if os.path.exists(HASH_MAP_FILE):
        with open(HASH_MAP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["string_to_id"], {int(k): v for k, v in data["id_to_string"].items()}
    return {}, {}

def get_area(decoded_edges):
    '''return the unique methods, classes and pakcages in the selected diff ranage'''
    u_packages = set()
    u_classes = set()
    u_methods = set()

    for edge in decoded_edges:
        package = edge[0].split('.')[:-2]
        u_packages.add('.'.join(package))
        u_classes.add(edge[0].split('.')[-2])
        u_methods.add(edge[0].split('.')[-1])

        package = edge[1].split('.')[:-2]
        u_packages.add('.'.join(package))
        u_classes.add(edge[1].split('.')[-2])
        u_methods.add(edge[1].split('.')[-1])
        

    u_mathods_list = list(u_methods)
    u_classes_list = list(u_classes)
    u_packages_list = list(u_packages)\
    
    json_data = json.dumps({'u_methods': u_mathods_list, 'u_classes': u_classes_list, 'u_package': u_packages_list})

    with open('../data/diff-sim/info/mt3vc1_area.json', 'w') as f:
        f.write(json_data)


def in_range(index, diff_range):
    '''checks if index is in range'''
    
    if index >= diff_range[0] and index <= diff_range[1]:
        return True
    return False


def decode_file(encoded_file, diff_range:tuple):
    """Decode an encoded file back to text format."""
    string_to_id, id_to_string = load_hash_map()
    decoded_edges = []

    encoded_file = encoded_file + '.encoded'
    encoded_file = os.path.join(ENCODED_TRACE_PATH, encoded_file)

    # Read encoded file and decode
    with open(encoded_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if in_range(index, diff_range):
                left, right = ast.literal_eval(line)
                decoded_edges.append((id_to_string[left], id_to_string[right]))

    
    get_area(decoded_edges)

    
    # decoded_file = encoded_file.replace(".encoded", ".decoded")
    # with open(decoded_file, "w", encoding="utf-8") as f:
    #     for edge in decoded_edges:
    #         f.write(f"{edge}\n")

    # print(f"Decoded {encoded_file} -> {decoded_file}")


def main():
    decode_file("VC1.txt", (110000, 116999))


if __name__ == '__main__':
    main()