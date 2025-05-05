# pool_embeddings.py

# graph as document


import os
import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# === Config ===
use_tfidf = True  # Set to False to use regular mean pooling
use_sum = False  # Set to True to use sum pooling

# === Paths ===
traces_path = '/20TB/mohammad/data/edge-traces-encode/new_cgs'
# walks_path = '/20TB/mohammad/data/cg_walks'
edge_repr_path = '/20TB/mohammad/data/cg_edge_repr'
embedding_model_path = '/20TB/mohammad/data/cg_embeddings/graph_word2vec.model'

if use_tfidf:
    embeddings_path = '/20TB/mohammad/data/cg_embeddings/sum/tfidf' if use_sum else '/20TB/mohammad/data/cg_embeddings/mean/tfidf'
else:
    embeddings_path = '/20TB/mohammad/data/cg_embeddings/sum/regular' if use_sum else '/20TB/mohammad/data/cg_embeddings/mean/regular'

os.makedirs(embeddings_path, exist_ok=True)

# === Load Word2Vec Model ===
print("Loading Word2Vec model...")
model = Word2Vec.load(embedding_model_path)


# === Helper to reconstruct graph nodes ===
def load_graph_nodes(program_name, edge_id):
    edge_repr_file = os.path.join(edge_repr_path, f'{program_name}_{edge_id}.edg')
    nodes = set()
    with open(edge_repr_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src, dst = parts[0], parts[1]
                nodes.add(src)
                nodes.add(dst)
    return list(nodes)

def get_id(program):
    edge_id_2_info = {}
    edge_file = os.path.join(traces_path, program, 'edges.csv')
    edges_df = pd.read_csv(edge_file)
    for _, row in edges_df.iterrows():
        edge_id = row['edge_id']
        method = row['method']
        offset = row['offset']
        target = row['target']
        edge_id_2_info[edge_id] = (method, offset, target)
    return edge_id_2_info

# === Main ===
def main():
    print("Pooling graph embeddings...")

    tfidf = None
    vocab = None
    edge_docs = []
    edge_lookup = []

    if use_tfidf:
        print("Building TF-IDF corpus...")
        for program_name in os.listdir(traces_path):
            program_folder = os.path.join(traces_path, program_name)
            if not os.path.isdir(program_folder):
                continue
            for edge_file in os.listdir(program_folder):
                if edge_file.endswith(".txt"):
                    edge_id = edge_file.split('.')[0]
                    node_ids = load_graph_nodes(program_name, edge_id)
                    edge_docs.append(" ".join(node_ids))
                    edge_lookup.append((program_name, edge_id))

        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(edge_docs)
        vocab = tfidf.vocabulary_

    for program_name in os.listdir(traces_path):
        program_folder = os.path.join(traces_path, program_name)
        if not os.path.isdir(program_folder):
            continue

        edge_id_2_info = get_id(program_name)
        all_rows = []

        for edge_file in os.listdir(program_folder):
            if not edge_file.endswith('.txt'):
                continue
            edge_id = edge_file.split('.')[0]
            node_ids = load_graph_nodes(program_name, edge_id)

            node_embs = []
            weights = []

            if use_tfidf:
                doc_idx = edge_lookup.index((program_name, edge_id)) if (program_name, edge_id) in edge_lookup else -1
                if doc_idx >= 0:
                    for node_id in node_ids:
                        if node_id in model.wv and node_id in vocab:
                            weight = tfidf_matrix[doc_idx, vocab[node_id]]
                            node_embs.append(model.wv[node_id] * weight)
                            weights.append(weight)
            else:
                for node_id in node_ids:
                    if node_id in model.wv:
                        node_embs.append(model.wv[node_id])

            if node_embs:
                node_embs = np.stack(node_embs)
                if use_tfidf and weights:
                    if use_sum:
                        graph_embedding = np.sum(node_embs, axis=0)
                    else:
                        weight_sum = np.sum(weights)
                        graph_embedding = np.sum(node_embs, axis=0) / weight_sum if weight_sum > 0 else np.mean(node_embs, axis=0)
                else:
                    if use_sum:
                        graph_embedding = np.sum(node_embs, axis=0)
                    else:
                        graph_embedding = np.mean(node_embs, axis=0)
            else:
                graph_embedding = np.zeros(model.vector_size)

            edge_id = int(edge_id)
            try:
                method, offset, target = edge_id_2_info[edge_id]
            except KeyError:
                print(f"Edge ID {edge_id} not found in edge_id_2_info for program {program_name}.")
                continue

            row = [method, offset, target] + list(graph_embedding)
            all_rows.append(row)

        if all_rows:
            embedding_dim = len(all_rows[0]) - 3
            columns = ['method', 'offset', 'target'] + [f'emb_{i}' for i in range(embedding_dim)]
            df = pd.DataFrame(all_rows, columns=columns)
            df.to_csv(os.path.join(embeddings_path, f'{program_name}.csv'), index=False)
            print(f"Saved embeddings for {program_name}.")

if __name__ == "__main__":
    main()
