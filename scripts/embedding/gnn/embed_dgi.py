import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import json
import random
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.models import DeepGraphInfomax

# Paths
traces_path = '/20TB/mohammad/data/edge-traces/new_cgs'
edge_repr_path = '/20TB/mohammad/data/cg_edge_repr'  # Precomputed edge files
embeddings_path = '/20TB/mohammad/data/cg_embeddings_dgi_weighted'
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
def load_graph_from_edgelist(args):
    program_name, edge_id = args
    path = os.path.join(edge_repr_path, f'{program_name}_{edge_id}.edg')
    if not os.path.exists(path):
        return None, None
    try:
        G = nx.read_weighted_edgelist(path, create_using=nx.DiGraph())
        if len(G) == 0:
            print(f"[SKIP] Empty graph: {program_name}_{edge_id}")
            return None, None

        for node in G.nodes():
            G.nodes[node]['x'] = [G.degree(node)]

        # Convert using torch_geometric.utils.from_networkx
        data = from_networkx(G)
        
        # Standardize edge weight: move from 'weight' to 'edge_weight'
        if hasattr(data, 'weight') and data.weight is not None:
            data.edge_weight = data.weight.float()
            del data.weight  # cleanup
        else:
            data.edge_weight = torch.ones(data.edge_index.shape[1])

        data.x = data.x.float()
        return data, (program_name, edge_id)

    except Exception as e:
        print(f"[EXCEPTION] Failed to load {program_name}_{edge_id}: {e}")
        return None, None


# === Main Training and Embedding Pipeline ===
def main(hidden_dim=64, epochs=20, max_workers=16, train_ratio=0.2):
    trace_files = []
    for program_name in os.listdir(traces_path):
        program_path = os.path.join(traces_path, program_name)
        if not os.path.isdir(program_path):
            continue
        for fname in os.listdir(program_path):
            if fname.endswith('.txt'):
                edge_id = fname.split('.')[0]
                trace_files.append((program_name, edge_id))

    print("Loading graphs from precomputed edge representations in parallel...")
    all_graphs = []
    all_metadata = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for graph_data, meta in tqdm(executor.map(load_graph_from_edgelist, trace_files), total=len(trace_files)):
            if graph_data is not None:
                all_graphs.append(graph_data)
                all_metadata.append(meta)

    print(f"✅ Loaded {len(all_graphs)} graphs")
    if not all_graphs:
        print("No graphs loaded. Exiting.")
        return

    # Step 2: Train DGI on a 20% subset
    sample_size = int(train_ratio * len(all_graphs))
    sampled_indices = random.sample(range(len(all_graphs)), sample_size)
    train_graphs = [all_graphs[i] for i in sampled_indices]

    in_dim = train_graphs[0].x.size(1)
    encoder = GCNEncoder(in_dim, hidden_dim).to(device)
    model = DeepGraphInfomax(
        hidden_channels=hidden_dim,
        encoder=encoder,
        summary=lambda z, *args: torch.sigmoid(z.mean(dim=0)),
        corruption=lambda x, edge_index, edge_weight=None: corruption(x, edge_index, edge_weight)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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


    # Step 3: Generate embeddings
    model.eval()
    program_to_rows = defaultdict(list)
    with torch.no_grad():
        for data, (program_name, edge_id) in tqdm(zip(all_graphs, all_metadata), desc="Embedding", total=len(all_graphs)):
            data = data.to(device)
            z = model.encoder(data.x, data.edge_index, data.edge_weight)
            batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            graph_embedding = global_mean_pool(z, batch=batch_vec).squeeze(0)
            row = [edge_id] + list(graph_embedding.cpu().numpy())
            program_to_rows[program_name].append(row)

    for program_name, rows in program_to_rows.items():
        columns = ['edge_id'] + [f'emb_{i}' for i in range(hidden_dim)]
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(os.path.join(embeddings_path, f'{program_name}.csv'), index=False)
        print(f"✅ Saved: {program_name} ({len(rows)} graphs)")

if __name__ == "__main__":
    main()
