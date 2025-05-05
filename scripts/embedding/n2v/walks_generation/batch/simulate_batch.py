# simulate_batch.py

import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import json
from collections import Counter
from pecanpy import pecanpy as node2vec
import concurrent.futures

# === Config ===
traces_path = '/20TB/mohammad/data/edge-traces-encode/new_cgs'
edge_repr_path = '/20TB/mohammad/data/cg_edge_repr'
walks_path = '/20TB/mohammad/data/cg_walks'

BATCH_SIZE = 500
MAX_WORKERS = 15  # Adjust based on CPU

os.makedirs(edge_repr_path, exist_ok=True)
os.makedirs(walks_path, exist_ok=True)


def load_trace_graph(program_name, edge_id, string_to_id, id_to_string):
    trace_path = os.path.join(traces_path, program_name, f'{edge_id}.txt')
    df = pd.read_csv(trace_path, names=['src', 'target'], skiprows=1)
    df.dropna(inplace=True)
    df['src'] = df['src'].astype(str)
    df['target'] = df['target'].astype(str)

    edge_counts = Counter(zip(df['src'], df['target']))

    G = nx.DiGraph()
    for (u_str, v_str), w in edge_counts.items():
        # u_id = map_method_to_id(u_str, string_to_id, id_to_string)
        # v_id = map_method_to_id(v_str, string_to_id, id_to_string)
        G.add_edge(u_str, v_str, weight=w)

    edge_repr_file = os.path.join(edge_repr_path, f'{program_name}_{edge_id}.edg')
    nx.write_weighted_edgelist(G, edge_repr_file, delimiter='\t')
    return G

def simulate_one_graph(program_name, edge_id, string_to_id, id_to_string):
    G = load_trace_graph(program_name, edge_id, string_to_id, id_to_string)
    edge_repr_file = os.path.join(edge_repr_path, f'{program_name}_{edge_id}.edg')
    
    g = node2vec.SparseOTF(p=1.0, q=1.0, verbose=False)
    g.read_edg(edge_repr_file, directed=True, weighted=True)
    # g.preprocess_transition_probs()
    walks = g.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]

    return walks

def main(batch_num):
    print(f"Simulating batch {batch_num}...")

    # === Load all graphs list ===
    graphs_file = os.path.join(walks_path, "all_graphs_list.json")
    with open(graphs_file, "r") as f:
        graph_edge_list = json.load(f)

    # # === Load hash maps ===
    # string_to_id, id_to_string = load_hash_map(hash_map_path)

    # === Pick this batch's graphs ===
    batch = graph_edge_list[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE]

    batch_walks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for program_name, edge_id in batch:
            futures.append(executor.submit(simulate_one_graph, program_name, edge_id, None, None))
            # futures.append(executor.submit(simulate_one_graph, program_name, edge_id, string_to_id, id_to_string))
        
        for future in concurrent.futures.as_completed(futures):
            walks = future.result()
            batch_walks.extend(walks)

    # === Save walks to file ===
    batch_file = os.path.join(walks_path, f'walks_batch_{batch_num}.txt')
    with open(batch_file, "w") as f:
        for walk in batch_walks:
            f.write(' '.join(walk) + '\n')

    print(f"Finished batch {batch_num}.")

if __name__ == "__main__":
    batch_num = int(sys.argv[1])  # receive batch number as CLI argument
    main(batch_num)
