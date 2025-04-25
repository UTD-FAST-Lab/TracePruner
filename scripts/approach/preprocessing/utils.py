import os
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
import torch

from collections import Counter


traces_path = '/home/mohammad/projects/CallGraphPruner/data/edge-traces/new_cgs'



def load_trace_graph(program_name, edge_id):
    """
    Converts a trace TXT (src,target per line) into a PyG-compatible graph.
    Repeated edges are converted to edge weights.
    """
    trace_path = os.path.join(traces_path, program_name, f'{edge_id}.txt')

    # Read the file, skip the first line
    df = pd.read_csv(trace_path, names=['src', 'target'], skiprows=1)

    # Drop any missing values (in case of bad lines)
    df.dropna(inplace=True)

    # Convert to strings (or ints, depending on trace format)
    df['src'] = df['src'].astype(str)
    df['target'] = df['target'].astype(str)

    # Count repeated edges for weight
    edge_counts = Counter(zip(df['src'], df['target']))

    # Create graph with weighted edges
    G = nx.DiGraph()
    for (u, v), w in edge_counts.items():
        G.add_edge(u, v, weight=w)

    return nx_to_pyg(G)


def nx_to_pyg(graph: nx.DiGraph) -> Data:
    """
    Converts a NetworkX DiGraph to PyG Data object with edge weights.
    Node features are one-hot encoded by default.
    """
    node_ids = list(graph.nodes)
    id_map = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    # Create identity node features (one-hot)
    x = torch.eye(num_nodes, dtype=torch.float)

    # Edge index and weights
    edges = [(id_map[u], id_map[v]) for u, v in graph.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
