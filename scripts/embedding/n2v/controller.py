# controller.py

import os
import subprocess
import json
import concurrent.futures


traces_path = '/20TB/mohammad/data/edge-traces-encode/new_cgs'
walks_path = '/20TB/mohammad/data/cg_walks'
BATCH_SIZE = 500
MAX_CONCURRENT_BATCHES = 10  # Adjust freely

os.makedirs(walks_path, exist_ok=True)


def launch_batch(batch_num):
    """Launch simulate_batch.py for a batch if not already done."""
    batch_file = os.path.join(walks_path, f'walks_batch_{batch_num}.txt')
    if os.path.exists(batch_file):
        print(f"Batch {batch_num} already completed. Skipping.")
        return

    print(f"Launching batch {batch_num}...")
    subprocess.run(["python", "simulate_batch.py", str(batch_num)], check=True)
    print(f"Finished batch {batch_num}.")


def main():
    # Step 1: collect all graphs
    graph_edge_list = []
    for program_name in os.listdir(traces_path):
        program_folder = os.path.join(traces_path, program_name)
        if not os.path.isdir(program_folder):
            continue
        for edge_file in os.listdir(program_folder):
            if edge_file.endswith('.txt'):
                edge_id = edge_file.split('.')[0]
                graph_edge_list.append((program_name, edge_id))

    print(f"Total graphs: {len(graph_edge_list)}")
    
    # Step 2: save full list
    graphs_file = os.path.join(walks_path, "all_graphs_list.json")
    with open(graphs_file, "w") as f:
        json.dump(graph_edge_list, f)

    # Step 3: Run simulate_batch.py per batch
    total_batches = (len(graph_edge_list) + BATCH_SIZE - 1) // BATCH_SIZE

        # Step 4: Launch missing batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_BATCHES) as executor:
        futures = []
        for batch_num in range(total_batches):
            batch_file = os.path.join(walks_path, f'walks_batch_{batch_num}.txt')
            if not os.path.exists(batch_file):
                futures.append(executor.submit(launch_batch, batch_num))
            else:
                print(f"Batch {batch_num} already exists, skipping.")

        concurrent.futures.wait(futures)

    print("All batches done.")

if __name__ == "__main__":
    main()
