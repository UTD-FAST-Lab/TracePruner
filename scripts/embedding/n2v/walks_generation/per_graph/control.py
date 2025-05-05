import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Config
edge_repr_path = '/20TB/mohammad/data/cg_edge_repr'
walks_output_path = '/20TB/mohammad/data/cg_walks_per_graph'
num_walks = 10
walk_length = 80
max_workers = 20  # You can tune this based on available memory/CPU

# Gather all tasks that haven't been completed yet
all_tasks = []

for fname in os.listdir(edge_repr_path):
    if fname.endswith('.edg'):
        parts = fname.replace('.edg', '').split('_')
        program_name = '_'.join(parts[:-1])
        edge_id = parts[-1]

        out_file = os.path.join(walks_output_path, program_name, f"{edge_id}.txt")
        if os.path.exists(out_file):
            continue

        all_tasks.append((program_name, edge_id))

print(f"Found {len(all_tasks)} graphs to process...")

# Run each graph in its own subprocess using a thread pool
def run_subprocess(program_name, edge_id):
    try:
        subprocess.run([
            'python3', 'run_one.py',
            program_name,
            edge_id,
            edge_repr_path,
            walks_output_path,
            str(num_walks),
            str(walk_length)
        ], check=True)
        return f"✅ {program_name}/{edge_id}"
    except subprocess.CalledProcessError as e:
        return f"❌ {program_name}/{edge_id} failed: {e}"

# Run in parallel
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_task = {
        executor.submit(run_subprocess, program_name, edge_id): (program_name, edge_id)
        for program_name, edge_id in all_tasks
    }

    for future in tqdm(as_completed(future_to_task), total=len(future_to_task)):
        result = future.result()
        if result.startswith("❌"):
            print(result)
