import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import json
import random
import pickle
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.models import DeepGraphInfomax

# Paths
FILTERED_GRAPH_DIR = '/20TB/mohammad/data/filtered_graphs'
embeddings_path = '/20TB/mohammad/data/cg_embeddings_dgi_weighted'
traces_path = '/20TB/mohammad/data/edge-traces-encode/new_cgs'

os.makedirs(embeddings_path, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Encoder for DGI (weighted)
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        return self.conv2(x, edge_index, edge_weight)

# Corruption function for DGI
def corruption(x, edge_index, edge_weight=None):
    return x[torch.randperm(x.size(0))], edge_index, edge_weight

# Load graph from precomputed edge file
# Load graph from precomputed edge file
def load_graph_from_pickle(file_path):
    try:

        parts = file_path.replace('.pickle', '').split('_')
        program_name = '_'.join(parts[:-1])
        edge_id = parts[-1]


        file_path = os.path.join(FILTERED_GRAPH_DIR, file_path)
        with open(file_path, 'rb') as f:
            G = pickle.load(f)

        # Use degree as the only node feature for both macro and regular nodes
        for node in G.nodes():
            G.nodes[node].clear()
            G.nodes[node]['x'] = [G.degree(node)]

        # Convert to PyG Data format
        data = from_networkx(G)

        # Standardize edge weight
        if hasattr(data, 'weight') and data.weight is not None:
            data.edge_weight = data.weight.float()
            del data.weight  # cleanup
        else:
            data.edge_weight = torch.ones(data.edge_index.shape[1])

        # Convert node features to tensor
        data.x = data.x.float()

        
        
        return (program_name, edge_id, data)

    except Exception as e:
        print(f"[EXCEPTION] Failed to load graph from {file_path}: {e}")
        return None


# === Training and Embedding Pipeline ===
def train_dgi(all_graphs, hidden_dim=64, epochs=20, train_ratio=0.2, lr=0.001):
    # Sample a subset of graphs for training
    sample_size = int(train_ratio * len(all_graphs))
    sampled_indices = random.sample(range(len(all_graphs)), sample_size)
    train_graphs = [all_graphs[i] for i in sampled_indices]

    # Determine the max feature length for padding
    max_feat_len = max([g.x.size(1) for g in train_graphs])

    # Pad node features to the same length
    for g in train_graphs:
        if g.x.size(1) < max_feat_len:
            padding = torch.zeros(g.num_nodes, max_feat_len - g.x.size(1))
            g.x = torch.cat([g.x, padding], dim=1)

    in_dim = max_feat_len
    encoder = GCNEncoder(in_dim, hidden_dim).to(device)
    model = DeepGraphInfomax(
        hidden_channels=hidden_dim,
        encoder=encoder,
        summary=lambda z, *args: torch.sigmoid(z.mean(dim=0)),
        corruption=lambda x, edge_index, edge_weight=None: corruption(x, edge_index, edge_weight)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training DGI on {len(train_graphs)} sampled graphs...")
    model.train()
    batch = Batch.from_data_list(train_graphs).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(batch.x, batch.edge_index, batch.edge_weight)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    return model, max_feat_len



def generate_embeddings(model, all_info, all_graphs, embeddings_path):
    model.eval()
    program_to_rows = defaultdict(list)

    with torch.no_grad():
        for i, data in enumerate(tqdm(all_graphs, desc="Embedding")):
            data = data.to(device)

            # Forward pass through the encoder
            z = model.encoder(data.x, data.edge_index, data.edge_weight)

            # Global pooling to get graph-level embedding
            batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            graph_embedding = global_mean_pool(z, batch=batch_vec).squeeze(0)

            # Extract metadata
            program_name, edge_id = all_info[i]
            edge_id_2_info = get_id(program_name)

            try:
                edge_id = int(edge_id)
                method, offset, target = edge_id_2_info[edge_id]
            except KeyError:
                print(f"⚠️ Edge ID {edge_id} not found in edge_id_2_info for program {program_name}. Skipping...")
                continue

            # Create the embedding row
            row = [method, offset, target] + graph_embedding.cpu().numpy().tolist()
            program_to_rows[program_name].append(row)

    # Save embeddings to CSV
    for program_name, rows in program_to_rows.items():
        columns = ['method', 'offset', 'target'] + [f'emb_{i}' for i in range(len(rows[0]) - 3)]
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(os.path.join(embeddings_path, f'{program_name}.csv'), index=False)
        print(f"✅ Saved: {program_name} ({len(rows)} graphs)")


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


# === Main Training and Embedding Pipeline ===
def main(hidden_dim=64, epochs=20, max_workers=16, lr=0.001 ,train_ratio=0.5):
    graph_files = sorted([f for f in os.listdir(FILTERED_GRAPH_DIR) if f.endswith('.pickle')])


    print("Loading graphs from precomputed pickle files in parallel...")
    all_graphs = []
    all_info = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for graph_data in tqdm(executor.map(load_graph_from_pickle, graph_files), total=len(graph_files)):
            if graph_data is not None:
                all_graphs.append(graph_data[2])
                all_info.append((graph_data[0], graph_data[1]))
    print(f"✅ Loaded {len(all_graphs)} graphs")
    if not all_graphs:
        print("No graphs loaded. Exiting.")
        return

    
    # Train the DGI model
    model, max_feat_len = train_dgi(all_graphs, hidden_dim=hidden_dim, epochs=epochs, train_ratio=train_ratio, lr=lr)

    # Generate embeddings
    generate_embeddings(model, all_info, all_graphs, embeddings_path)

if __name__ == "__main__":
    main()
