# run_one.py
import sys
import os
from pecanpy import pecanpy as node2vec

program_name = sys.argv[1]
edge_id = sys.argv[2]
edge_repr_path = sys.argv[3]
walks_output_path = sys.argv[4]
num_walks = int(sys.argv[5])
walk_length = int(sys.argv[6])

path = os.path.join(edge_repr_path, f'{program_name}_{edge_id}.edg')
g = node2vec.SparseOTF(p=1.0, q=1.0, verbose=False)
g.read_edg(path, directed=True, weighted=True)

walks = g.simulate_walks(num_walks=num_walks, walk_length=walk_length)

out_dir = os.path.join(walks_output_path, program_name)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{edge_id}.txt")
with open(out_path, "w") as f:
    for walk in walks:
        f.write(" ".join(map(str, walk)) + "\n")
