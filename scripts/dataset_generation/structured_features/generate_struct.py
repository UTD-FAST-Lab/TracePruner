import pandas as pd
import os
from functools import reduce
import csv
import networkx as nx
from collections import defaultdict

dataset_dir = "/20TB/mohammad/xcorpus-total-recall/dataset"
output_dir = '/20TB/mohammad/xcorpus-total-recall/features/struct' 

programs = [
    'axion',
    'batik',
    'xerces',
    'jasml'
]

tools = [
    'wala',
    'doop',
    'opal'
]

selected_keys = [
    ('wala', 'v1_19'), #0cfa,ONE_FLOW_TO_CASTS_NO_METHOD_INVOKE (excluding xerces)
    ('wala', 'v3_0'),  #1cfa,ONE_FLOW_TO_CASTS_NO_METHOD_INVOKE (excluding xerces)
    ('wala', 'v1_23'),  #0cfa,String_only (excluding xerces)
    ('doop', 'v1_39'),  #0cfa_on (excluding jasml)
    ('doop', 'v3_5'),   #1_type, on
    ('doop', 'v2_0'),  # 1obj,off (excluding xerces)
    ('opal', 'v1_0'),   #cha
    ('opal', 'v1_8'),   #0-1cfa
]


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

    for program in programs:
        total_unknowns_dict = {}

        for tool in tools:
            tool_dir = os.path.join(dataset_dir, tool, 'without_jdk', program)
            if not os.path.exists(tool_dir):
                continue

            for config in os.listdir(tool_dir):
                config_dir = os.path.join(tool_dir, config)
                if not os.path.isdir(config_dir):
                    continue

                # Read the unknowns file
                unknown_file = os.path.join(config_dir, 'total_edges.csv')
                if not os.path.exists(unknown_file):
                    continue

                unknown_df = pd.read_csv(unknown_file).drop_duplicates(subset=['method', 'offset', 'target'])

                if unknown_df.empty:
                    continue

                total_unknowns_dict[(tool, config)] = unknown_df

        for key, value in total_unknowns_dict.items():
            if key in selected_keys:
                edges = [(row['method'], row['offset'], row['target']) for _, row in value.iterrows()]
                main_method = 'Entrypoint.entrypoint:(Ljava/lang/String;)V'
                G = build_graph(edges)
                if main_method not in G.nodes:
                    main_method = 'Entrypoint.entrypoint:(Ljava/io/File;)V'
                if main_method not in G.nodes:
                    continue

                features = compute_features(G, edges, main=main_method)
                output_path = os.path.join(output_dir, program, f"struct_{key[0]}_{key[1]}.csv")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                write_output(output_path, features)




if __name__ == '__main__':
    main()