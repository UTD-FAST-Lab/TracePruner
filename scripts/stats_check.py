import os
import pandas as pd

data_dir = '/home/mohammad/projects/CallGraphPruner_data/dataset-high-precision-callgraphs/full_callgraphs_set'

transitive = 0
direct = 0

true_trans = 0
false_trans = 0

true_direct = 0
false_direct = 0

for dir in os.listdir(data_dir):
    if not dir.startswith('.'):
        csv_path = os.path.join(data_dir, dir, 'wala0cfa.csv')

        if not os.path.exists(csv_path):
            print(f"Skipping {dir}, wala0.csv not found.")
            continue

        df = pd.read_csv(csv_path)

        # Remove duplicates based on method, offset, target
        df = df.drop_duplicates(subset=['method', 'offset', 'target'])

        # Count total transitive and direct edges
        transitive += df['wala-cge-0cfa-noreflect-intf-trans'].sum()
        direct += df['wala-cge-0cfa-noreflect-intf-direct'].sum()

        # Count true/false transitive edges
        true_trans += df[(df['wiretap'] == 1) & (df['wala-cge-0cfa-noreflect-intf-trans'] == 1)].shape[0]
        false_trans += df[(df['wiretap'] == 0) & (df['wala-cge-0cfa-noreflect-intf-trans'] == 1)].shape[0]

        # Count true/false direct edges
        true_direct += df[(df['wiretap'] == 1) & (df['wala-cge-0cfa-noreflect-intf-direct'] == 1)].shape[0]
        false_direct += df[(df['wiretap'] == 0) & (df['wala-cge-0cfa-noreflect-intf-direct'] == 1)].shape[0]

# Print summary
print(f"Total transitive edges: {transitive}")
print(f"Total direct edges: {direct}")
print(f"True transitive edges: {true_trans}")
print(f"False transitive edges: {false_trans}")
print(f"True direct edges: {true_direct}")
print(f"False direct edges: {false_direct}")