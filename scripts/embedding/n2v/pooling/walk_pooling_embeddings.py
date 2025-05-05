# tfidf_walk_pooling.py

import os
import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

use_sum = True  # Set to True to use sum pooling

# === Paths ===
traces_path = '/20TB/mohammad/data/edge-traces-encode/new_cgs'
walks_path = '/20TB/mohammad/data/cg_walks_per_graph' 
edge_repr_path = '/20TB/mohammad/data/cg_edge_repr'
embedding_model_path = '/20TB/mohammad/data/cg_embeddings_per_graph/graph_word2vec.model'
embeddings_path = '/20TB/mohammad/data/cg_embeddings_per_graph/sum/tfidf' if use_sum else '/20TB/mohammad/data/cg_embeddings_per_graph/mean/tfidf'

os.makedirs(embeddings_path, exist_ok=True)

# === Load Word2Vec Model ===
print("Loading Word2Vec model...")
model = Word2Vec.load(embedding_model_path)


# === Load edge ID → method/offset/target ===
def get_id(program):
    edge_id_2_info = {}
    edge_file = os.path.join(traces_path, program, 'edges.csv')
    edges_df = pd.read_csv(edge_file)
    for _, row in edges_df.iterrows():
        edge_id = row['edge_id']
        method = row['method']
        offset = row['offset']
        target = row['target']
        edge_id_2_info[str(edge_id)] = (method, offset, target)
    return edge_id_2_info

# === Load walks for a graph ===
def load_walks(program_name, edge_id):
    walk_file = os.path.join(walks_path, program_name, f'{edge_id}.txt')
    if not os.path.exists(walk_file):
        return []
    with open(walk_file, "r") as f:
        walks = [line.strip().split() for line in f if line.strip()]
    return walks

# === Main ===
def main():
    print("Building TF-IDF weighted embeddings from walks...")

    all_docs = []  # list of str
    edge_lookup = []

    # Step 1: collect walks as documents
    for program_name in os.listdir(walks_path):
        program_folder = os.path.join(walks_path, program_name)
        if not os.path.isdir(program_folder):
            continue
        for fname in os.listdir(program_folder):
            if fname.endswith(".txt"):
                edge_id = fname.split('.')[0]
                walks = load_walks(program_name, edge_id)
                doc = [' '.join(w) for w in walks]
                doc_flat = ' '.join(doc)  # merge all walks into one document
                all_docs.append(doc_flat)
                edge_lookup.append((program_name, edge_id))

    print(f"Total walk-documents: {len(all_docs)}")

    # Step 2: TF-IDF fitting
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(all_docs)
    vocab = tfidf.vocabulary_  # maps node str -> column idx

    # Step 3: pool embeddings using TF-IDF
    program_to_rows = defaultdict(list)
    for i, (program_name, edge_id) in enumerate(edge_lookup):
        node_weights = {}
        for token in all_docs[i].split():
            if token in vocab:
                col_idx = vocab[token]
                tfidf_value = tfidf_matrix[i, col_idx]
                node_weights[token] = node_weights.get(token, 0) + tfidf_value

        # Weighted pooling
        node_embs = []
        weights = []
        for node, w in node_weights.items():
            if node in model.wv:
                node_embs.append(model.wv[node] * w)
                weights.append(w)

        if node_embs:
            node_embs = np.stack(node_embs)
            if use_sum:
                graph_embedding = np.sum(node_embs, axis=0)
            else:
                weight_sum = np.sum(weights)
                graph_embedding = np.sum(node_embs, axis=0) / weight_sum if weight_sum > 0 else np.mean(node_embs, axis=0)
        else:
            graph_embedding = np.zeros(model.vector_size)

        # Save row
        edge_id_int = int(edge_id)
        edge_id_2_info = get_id(program_name)
        if edge_id not in edge_id_2_info:
            continue
        method, offset, target = edge_id_2_info[edge_id]
        row = [method, offset, target] + list(graph_embedding)
        program_to_rows[program_name].append(row)

    # Step 4: save per program
    for program_name, rows in program_to_rows.items():
        if rows:
            dim = len(rows[0]) - 3
            columns = ['method', 'offset', 'target'] + [f'emb_{i}' for i in range(dim)]
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(os.path.join(embeddings_path, f'{program_name}.csv'), index=False)
            print(f"✅ Saved: {program_name} ({len(rows)} graphs)")

if __name__ == "__main__":
    main()