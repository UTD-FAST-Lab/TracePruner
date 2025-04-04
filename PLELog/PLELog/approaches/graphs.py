import networkx as nx
import numpy as np
from karateclub import Node2Vec, Graph2Vec
from tqdm import tqdm


def embed_graphs_node2vec(graphs, dimensions=128):
    """Embeds each graph using Node2Vec (karateclub) by averaging node embeddings."""
    embeddings = []

    for G in tqdm(graphs, desc="Embedding with Node2Vec"):
        if len(G.nodes) == 0:
            embeddings.append(np.zeros(dimensions))
            continue

        # Add dummy label for compatibility
        G_copy = G.copy()
        nx.set_node_attributes(G_copy, 1, "label")

        # Relabel nodes to 0...N-1
        G_reindexed = nx.convert_node_labels_to_integers(G_copy)

        model = Node2Vec(dimensions=dimensions)
        model.fit(G_reindexed)

        node_embeds = model.get_embedding()
        graph_embed = np.mean(node_embeds, axis=0)
        embeddings.append(graph_embed)

    return embeddings


def embed_graphs_graph2vec(graphs, dimensions=300, wl_iterations=2):
    """Embeds each graph using Graph2Vec."""
    graphs_processed = []
    for G in graphs:
        G_copy = G.copy()
        nx.set_node_attributes(G_copy, 1, "label")  # dummy label
        # Relabel nodes to 0...N-1
        G_reindexed = nx.convert_node_labels_to_integers(G_copy)
        graphs_processed.append(G_reindexed)

    model = Graph2Vec(dimensions=dimensions, wl_iterations=wl_iterations)
    model.fit(graphs_processed)
    return model.get_embedding()
