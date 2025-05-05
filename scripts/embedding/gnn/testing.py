import os
import pandas as pd

traces_path = '/20TB/mohammad/data/edge-traces/new_cgs'
embeddings_path = '/20TB/mohammad/data/cg_embeddings_dgi_weighted'


def get_id(program):
    '''Read the edges.csv file and map edge_id -> (method, offset, target)'''
    edge_file = os.path.join(traces_path, program, 'edges.csv')
    edges_df = pd.read_csv(edge_file)
    return {
        row['edge_id']: (row['method'], row['offset'], row['target'])
        for _, row in edges_df.iterrows()
    }


def main():
    for program_name in os.listdir(traces_path):
        edge_info = get_id(program_name)

        embedding_file = os.path.join(embeddings_path, f'{program_name}.csv')
        if not os.path.exists(embedding_file):
            print(f"Missing: {embedding_file}")
            continue

        un_df = pd.read_csv(embedding_file)

        updated_rows = []

        for _, row in un_df.iterrows():
            edge_id = row['edge_id']
            if edge_id in edge_info:
                method, offset, target = edge_info[edge_id]
                row['method'] = method
                row['offset'] = offset
                row['target'] = target
                updated_rows.append(row)
            else:
                print(f"[{program_name}] Skipping edge ID {edge_id} (not found in edges.csv)")

        # Create new filtered DataFrame and save
        filtered_df = pd.DataFrame(updated_rows)
        filtered_df.to_csv(embedding_file, index=False)


main()
