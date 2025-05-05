import os
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from pecanpy import pecanpy as node2vec
import gc

# === Config ===
edge_repr_path = '/20TB/mohammad/data/cg_edge_repr'
walks_output_path = '/20TB/mohammad/data/cg_walks_per_graph'
num_walks = 10
walk_length = 80
num_threads = 20

os.makedirs(walks_output_path, exist_ok=True)

# === Utility to read a graph from .edg file ===
def load_graph(program_name, edge_id):
    path = os.path.join(edge_repr_path, f'{program_name}_{edge_id}.edg')
    if not os.path.exists(path):
        return None, None
    return path, (program_name, edge_id)

# === Generate walks for one graph ===
def process_graph(args):
    path, (program_name, edge_id) = args
    try:
        # Load into SparseOTF
        g = node2vec.SparseOTF(p=1.0, q=1.0, verbose=False)
        g.read_edg(path, directed=True, weighted=True)

        # Generate walks
        walks = g.simulate_walks(num_walks=num_walks, walk_length=walk_length)

        # Save walks to file in appropriate folder
        out_dir = os.path.join(walks_output_path, program_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{edge_id}.txt")
        with open(out_path, "w") as f:
            for walk in walks:
                f.write(" ".join(map(str, walk)) + "\n")

        # Clean up
        del g, walks
        gc.collect()
        
        return f"✅ {program_name}/{edge_id}"

    except Exception as e:
        return f"❌ {program_name}/{edge_id} failed: {e}"

# === Main ===
def main():
    all_args = []
    for fname in os.listdir(edge_repr_path):
        if fname.endswith('.edg'):
            parts = fname.replace('.edg', '').split('_')
            program_name = '_'.join(parts[:-1])
            edge_id = parts[-1]
            path = os.path.join(edge_repr_path, fname)

            # Check if output already exists
            out_file = os.path.join(walks_output_path, program_name, f"{edge_id}.txt")
            if os.path.exists(out_file):
                continue  # Skip this one

            all_args.append((path, (program_name, edge_id)))

    print(f"Found {len(all_args)} graphs to process (excluding completed ones). Generating walks...")

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_graph, arg) for arg in all_args]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result.startswith("❌"):
                print(result)

if __name__ == "__main__":
    main()
