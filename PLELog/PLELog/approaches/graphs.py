import networkx as nx
import numpy as np
from tqdm import tqdm
# from karateclub import Node2Vec, Graph2Vec
from node2vec import Node2Vec


def embed_graphs_node2vec_original(graphs, dimensions=128, walk_length=20, num_walks=100, workers=2, window=5, directed=True):
    
    g_embeddings = []

    for i, G in enumerate(tqdm(graphs, desc= 'embedding graphs with node2vec')):
    
        # Add dummy weights if missing
        for u, v in G.edges():
            if 'weight' not in G[u][v]:
                print("no weight !!!")
                G[u][v]['weight'] = 1.0

        # Initialize Node2Vec
        node2vec = Node2Vec(
            G,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            weight_key='weight',  # this enables weighted walks
            quiet=True  # suppress tqdm inside node2vec
        )

        model = node2vec.fit(window=window, min_count=1)

        # Get node embeddings
        node_ids = list(G.nodes())
        node_embeddings = np.array([model.wv[str(n)] for n in node_ids if str(n) in model.wv])

        # Aggregate node embeddings (mean pooling)
        if len(node_embeddings) == 0:
            return np.zeros(dimensions)
        graph_embedding = np.mean(node_embeddings, axis=0)

        g_embeddings.append(graph_embedding)

    return g_embeddings
    

# def embed_graphs_node2vec(graphs, dimensions=128):
#     """Embeds each graph using Node2Vec (karateclub) by averaging node embeddings."""
#     embeddings = []

#     for G in tqdm(graphs, desc="Embedding with Node2Vec"):
#         if len(G.nodes) == 0:
#             embeddings.append(np.zeros(dimensions))
#             continue

#         # Add dummy label for compatibility
#         G_copy = G.copy()
#         nx.set_node_attributes(G_copy, 1, "label")

#         # Relabel nodes to 0...N-1
#         G_reindexed = nx.convert_node_labels_to_integers(G_copy)

#         model = Node2Vec(dimensions=dimensions)
#         model.fit(G_reindexed)

#         node_embeds = model.get_embedding()
#         graph_embed = np.mean(node_embeds, axis=0)
#         embeddings.append(graph_embed)

#     return embeddings


# def embed_graphs_graph2vec(graphs, dimensions=128, wl_iterations=2):
#     """Embeds each graph using Graph2Vec."""
#     graphs_processed = []
#     for G in graphs:
#         G_copy = G.copy()
#         nx.set_node_attributes(G_copy, 1, "label")  # dummy label
#         # Relabel nodes to 0...N-1
#         G_reindexed = nx.convert_node_labels_to_integers(G_copy)
#         graphs_processed.append(G_reindexed)

#     model = Graph2Vec(dimensions=dimensions, wl_iterations=wl_iterations)
#     model.fit(graphs_processed)
#     return model.get_embedding()
