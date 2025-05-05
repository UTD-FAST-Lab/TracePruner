# check_embeddings_report.py

import os
import pandas as pd
import numpy as np

# === Paths ===
embeddings_path = '/20TB/mohammad/data/cg_embeddings'

# === Tracking stats
total_graphs = 0
zero_graphs = 0
normal_graphs = 0

program_reports = []

def is_all_zero(vec):
    return np.allclose(vec, 0.0)

# === Process all program embedding files
for filename in os.listdir(embeddings_path):
    if not filename.endswith('.csv'):
        continue

    filepath = os.path.join(embeddings_path, filename)
    df = pd.read_csv(filepath)

    for idx, row in df.iterrows():
        emb = row.values[1:]  # Skip edge_id
        total_graphs += 1
        if is_all_zero(emb):
            zero_graphs += 1
        else:
            normal_graphs += 1

    program_reports.append((filename, len(df)))

# === Print results
print(f"Total graphs: {total_graphs}")
print(f"Graphs with all zero embedding: {zero_graphs} ({100 * zero_graphs / total_graphs:.2f}%)")
print(f"Graphs with valid embeddings: {normal_graphs} ({100 * normal_graphs / total_graphs:.2f}%)")

print("\nGraphs per program:")
for program, count in program_reports:
    print(f"  {program}: {count} graphs")

