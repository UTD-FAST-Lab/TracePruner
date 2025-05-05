from karateclub import Graph2Vec

# GL2Vec model
model = Graph2Vec(
    dimensions=128,
    wl_iterations=2,
)

# Fit model
model.fit([G])

# Get graph embeddings
graph_embedding = model.get_embedding()
print(graph_embedding.shape)  # (1, 128)
