# train_word2vec.py

import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from itertools import chain

# === Config ===
use_batch = False  # Set to False if you want to use all walks

# === Paths ===
if use_batch:
    walks_path = '/20TB/mohammad/data/cg_walks'
    embedding_model_path = '/20TB/mohammad/data/cg_embeddings/graph_word2vec.model'
else:
    walks_path = '/20TB/mohammad/data/cg_walks_per_graph'
    embedding_model_path = '/20TB/mohammad/data/cg_embeddings_per_graph/graph_word2vec.model'

os.makedirs(os.path.dirname(embedding_model_path), exist_ok=True)

# Step 1: Collect walk files
walk_files = []
if use_batch:
    walk_files = sorted([
        os.path.join(walks_path, f)
        for f in os.listdir(walks_path)
        if f.startswith('walks_batch')
    ])
else:
    for root, _, files in os.walk(walks_path):
        for f in files:
            if f.endswith('.txt'):
                walk_files.append(os.path.join(root, f))

# Step 2: Create streaming corpus
sentences = [LineSentence(walk_file) for walk_file in walk_files]
streaming_sentences = chain(*sentences)  # Flatten multiple LineSentence iterators

# Step 3: Train Word2Vec
print(f"Training Word2Vec on {len(walk_files)} walk files...")

w2v_model = Word2Vec(
    sentences=streaming_sentences,
    vector_size=128,
    window=10,
    min_count=0,
    sg=1,
    workers=48,
    epochs=5
)

# Step 4: Save model
w2v_model.save(embedding_model_path)
print(f"Saved Word2Vec model to {embedding_model_path}")