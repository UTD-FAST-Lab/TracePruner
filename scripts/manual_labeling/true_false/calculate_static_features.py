import csv
import networkx as nx
from collections import defaultdict
from pathlib import Path

# Paths
program_list_file = '/home/mohammad/projects/CallGraphPruner/data/programs/all_programs.txt'  # Path to the list of program names
base_directory = Path('/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9')




def read_callgraph(file_path):
    edges = []
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            target = row['target']
            if method.startswith("java/") or target.startswith("java/"):
                continue
            edges.append((method, row['offset'], target))
    return edges

def build_graph(edges):
    G = nx.DiGraph()
    for method, offset, target in edges:
        G.add_edge(method, target, offset=offset)
    return G

def shortest_path_length_from_main(G, main="<boot>"):
    try:
        return nx.single_source_shortest_path_length(G, main)
    except nx.NetworkXError:
        return {}

def compute_features(G, edges, main="<boot>"):
    edge_count = G.number_of_edges()
    node_count = G.number_of_nodes()
    avg_degree = sum(dict(G.out_degree()).values()) / node_count if node_count else 0

    lfanout_map = defaultdict(int)
    repeated_edge_map = defaultdict(int)
    for u, v, data in G.edges(data=True):
        callsite = (u, data['offset'])
        lfanout_map[callsite] += 1
        repeated_edge_map[(u, v)] += 1
    avg_lfanout = sum(lfanout_map.values()) / len(lfanout_map) if lfanout_map else 0

    path_lengths = shortest_path_length_from_main(G, main)

    results = []
    for method, offset, target in edges:
        src_in_deg = G.in_degree(method)
        src_out_deg = G.out_degree(method)
        tgt_in_deg = G.in_degree(target)
        tgt_out_deg = G.out_degree(target)
        depth = path_lengths.get(method, -1)
        repeated = repeated_edge_map[(method, target)]
        lfanout = lfanout_map[(method, offset)]

        results.append({
            "method": method,
            "offset": offset,
            "target": target,
            "src-node-in-deg": src_in_deg,
            "src-node-out-deg": src_out_deg,
            "dest-node-in-deg": tgt_in_deg,
            "dest-node-out-deg": tgt_out_deg,
            "depth": depth,
            "repeated-edges": repeated,
            "L-fanout": lfanout,
            "node-count": node_count,
            "edge-count": edge_count,
            "avg-degree": avg_degree,
            "avg-L-fanout": avg_lfanout
        })
    return results

def write_output(output_path, data):
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def main():

    # Read list of programs
    with open(program_list_file, 'r') as file:
        program_names = [line.strip() for line in file if line.strip()]

    for program in program_names:

        input_csv = base_directory / program / 'wala0cfa_filtered.csv'
        output_dir = base_directory / program / 'static_featuers'
        output_dir.mkdir(exist_ok=True)
        output_csv = output_dir / 'wala0cfa_filtered.csv'

        edges = read_callgraph(input_csv)
        G = build_graph(edges)
        features = compute_features(G, edges)
        write_output(output_csv, features)

if __name__ == "__main__":
    main()
