
import os
import pandas as pd
import numpy as np
import networkx as nx
import json
from collections import Counter, defaultdict
import concurrent.futures
from pecanpy import pecanpy as node2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


# === Paths ===
traces_path = '/20TB/mohammad/data/edge-traces/new_cgs'
edge_repr_path = '/20TB/mohammad/data/cg_edge_repr'
embeddings_path = '/20TB/mohammad/data/cg_embeddings'
hash_map_path = '/home/mohammad/projects/TracePruner/data/WALA_hash_map.json'
walks_path = '/20TB/mohammad/data/cg_walks'   # NEW folder to store walks

os.makedirs(walks_path, exist_ok=True)
os.makedirs(edge_repr_path, exist_ok=True)
os.makedirs(embeddings_path, exist_ok=True)

# === Functions ===

def load_hash_map(path):
    """Load existing hash map from file, or create a new one."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["string_to_id"], {int(k): v for k, v in data["id_to_string"].items()}
    print("Hash map not found. Starting fresh.")
    return {}, {}

def save_hash_map(path, string_to_id, id_to_string):
    """Save hash map to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"string_to_id": string_to_id, "id_to_string": id_to_string}, f, indent=4)

def map_method_to_id(method, string_to_id, id_to_string):
    """Map method name to unique ID, adding if not already present."""
    if method not in string_to_id:
        new_id = len(string_to_id)
        string_to_id[method] = new_id + 1
        id_to_string[new_id] = method
    return string_to_id[method]

def load_trace_graph(program_name, edge_id, string_to_id, id_to_string):
    """
    Load a trace file and build a directed weighted graph based on method mappings.
    Update string_to_id, id_to_string as needed.
    """
    trace_path = os.path.join(traces_path, program_name, f'{edge_id}.txt')

    df = pd.read_csv(trace_path, names=['src', 'target'], skiprows=1)
    df.dropna(inplace=True)
    df['src'] = df['src'].astype(str)
    df['target'] = df['target'].astype(str)

    edge_counts = Counter(zip(df['src'], df['target']))

    G = nx.DiGraph()
    for (u_str, v_str), w in edge_counts.items():
        u_id = map_method_to_id(u_str, string_to_id, id_to_string)
        v_id = map_method_to_id(v_str, string_to_id, id_to_string)
        G.add_edge(u_id, v_id, weight=w)

    # Save graph for Node2Vec
    edge_repr_file = os.path.join(edge_repr_path, f'{program_name}_{edge_id}.edg')
    nx.write_weighted_edgelist(G, edge_repr_file, delimiter='\t')

    return G

# def simulate_walks_for_graph(edge_repr_file):
#     """
#     Simulate Node2Vec walks for a graph from its edge list file.
#     """
#     g = node2vec.SparseOTF(p=1.0, q=1.0, verbose=False)
#     g.read_edg(edge_repr_file, directed=True, weighted=True)
#     # g.preprocess_transition_probs()
#     walks = g.simulate_walks(num_walks=10, walk_length=80)
#     walks = [list(map(str, walk)) for walk in walks]
#     return walks

# def main():
#     string_to_id, id_to_string = load_hash_map(hash_map_path)
#     all_walks = []  # collect all walks from all graphs
#     edge_to_nodes = {}  # map edge_id -> list of nodes (IDs) in its graph

#     # Step 1: Build all walks and node mappings
#     for program_name in os.listdir(traces_path):
#         program_folder = os.path.join(traces_path, program_name)
#         if not os.path.isdir(program_folder):
#             continue

#         for edge_file in os.listdir(program_folder):
#             if not edge_file.endswith('.txt'):
#                 continue
#             edge_id = edge_file.split('.')[0]

#             # Load graph and update mapping
#             G = load_trace_graph(program_name, edge_id, string_to_id, id_to_string)
#             edge_repr_file = os.path.join(edge_repr_path, f'{program_name}_{edge_id}.edg')

#             # Simulate walks
#             walks = simulate_walks_for_graph(edge_repr_file)

#             walks = [list(map(str, walk)) for walk in walks]
#             all_walks.extend(walks)

#             # Save nodes for pooling later
#             edge_to_nodes[f'{program_name}_{edge_id}'] = list(G.nodes())

#     # Save the updated hash map
#     save_hash_map(hash_map_path, string_to_id, id_to_string)

#     # Step 2: Train Word2Vec on all walks together
#     print(f"Training Word2Vec on {len(all_walks)} walks...")
#     w2v_model = Word2Vec(
#         sentences=all_walks,
#         vector_size=128,
#         window=10,
#         min_count=0,
#         sg=1,
#         workers=8,
#         epochs=5
#     )
#     print("Training complete.")

#     # Step 3: Pool node embeddings to graph embeddings
#     for program_name in os.listdir(traces_path):
#         program_folder = os.path.join(traces_path, program_name)
#         if not os.path.isdir(program_folder):
#             continue

#         all_rows = []

#         for edge_file in os.listdir(program_folder):
#             if not edge_file.endswith('.txt'):
#                 continue
#             edge_id = edge_file.split('.')[0]
#             full_edge_id = f'{program_name}_{edge_id}'

#             if full_edge_id not in edge_to_nodes:
#                 continue

#             node_embs = []
#             for n in edge_to_nodes[full_edge_id]:
#                 str_n = str(n)  # because Word2Vec keys are strings
#                 if str_n in w2v_model.wv:
#                     node_embs.append(w2v_model.wv[str_n])

#             if node_embs:
#                 node_embs = np.stack(node_embs)
#                 graph_embedding = np.mean(node_embs, axis=0)
#             else:
#                 graph_embedding = np.zeros(128)  # fallback if no nodes

#             row = [edge_id] + list(graph_embedding)
#             all_rows.append(row)

#         # Save per program
#         if all_rows:
#             embedding_dim = len(all_rows[0]) - 1
#             columns = ['edge_id'] + [f'emb_{i}' for i in range(embedding_dim)]
#             df = pd.DataFrame(all_rows, columns=columns)
#             df.to_csv(os.path.join(embeddings_path, f'{program_name}.csv'), index=False)
#             print(f"Saved embeddings for {program_name}.")

def simulate_walks_for_graph(edge_repr_file):
 

    g = node2vec.SparseOTF(p=1.0, q=1.0, verbose=False)
    g.read_edg(edge_repr_file, directed=True, weighted=True)
    # g.preprocess_transition_probs()

    walks = g.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]  # Convert node IDs to strings

    return walks

def main():
    string_to_id, id_to_string = load_hash_map(hash_map_path)

    # all_walks = []
    edge_to_nodes = {}

    # === Collect all graph edge IDs first ===
    graph_edge_list = []
    for program_name in os.listdir(traces_path):
        program_folder = os.path.join(traces_path, program_name)
        if not os.path.isdir(program_folder):
            continue

        for edge_file in os.listdir(program_folder):
            if edge_file.endswith('.txt'):
                edge_id = edge_file.split('.')[0]
                graph_edge_list.append((program_name, edge_id))

    # === Batch simulate walks ===
    print(f"Simulating walks for {len(graph_edge_list)} graphs...")

    batch_size = 1000
    walk_files = []

    for batch_start in range(0, len(graph_edge_list), batch_size):
        batch = graph_edge_list[batch_start:batch_start+batch_size]
        batch_walks = []
        print(f"Processing batch {batch_start // batch_size + 1}...")

        for program_name, edge_id in batch:
            G = load_trace_graph(program_name, edge_id, string_to_id, id_to_string)
            edge_repr_file = os.path.join(edge_repr_path, f'{program_name}_{edge_id}.edg')

            walks = simulate_walks_for_graph(edge_repr_file)
            batch_walks.extend(walks)

            full_edge_id = f'{program_name}_{edge_id}'
            edge_to_nodes[full_edge_id] = list(G.nodes())

        # Save batch of walks to a file
        batch_file = os.path.join(walks_path, f'walks_batch_{batch_start // batch_size + 1}.txt')
        walk_files.append(batch_file)

        with open(batch_file, "w") as f:
            for walk in batch_walks:
                f.write(' '.join(walk) + '\n')

    # === Save updated hash map ===
    save_hash_map(hash_map_path, string_to_id, id_to_string)



    # === Now train Word2Vec over all walk files ===
    print("Training Word2Vec...")
    sentences = []
    for walk_file in walk_files:
        sentences.append(LineSentence(walk_file))  # LineSentence supports lazy loading

    # Flatten multiple LineSentence objects
    from itertools import chain
    sentences = chain(*sentences)

    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=128,
        window=10,
        min_count=0,
        sg=1,
        workers=8,
        epochs=5
    )
    print("Training complete.")

    # === Pool node embeddings ===
    print("Pooling graph embeddings...")
    program_to_rows = defaultdict(list)

    for full_edge_id, nodes in edge_to_nodes.items():
        program_name, edge_id = full_edge_id.split('_', 1)
        node_embs = []

        for n in nodes:
            str_n = str(n)
            if str_n in w2v_model.wv:
                node_embs.append(w2v_model.wv[str_n])

        if node_embs:
            node_embs = np.stack(node_embs)
            graph_embedding = np.mean(node_embs, axis=0)
        else:
            graph_embedding = np.zeros(128)

        row = [edge_id] + list(graph_embedding)
        program_to_rows[program_name].append(row)

    # === Save per-program CSVs ===
    for program_name, rows in program_to_rows.items():
        embedding_dim = len(rows[0]) - 1
        columns = ['edge_id'] + [f'emb_{i}' for i in range(embedding_dim)]
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(os.path.join(embeddings_path, f'{program_name}.csv'), index=False)
        print(f"Saved embeddings for {program_name}.")

if __name__ == "__main__":
    main()



