import json
import pandas as pd
import os

var_repr_dir = "/20TB/mohammad/data/var_repr/"
var_embeddings_dir = "/20TB/mohammad/data/var_embeddings/"

# Create directories if they do not exist
os.makedirs(var_embeddings_dir, exist_ok=True)

def parse_points_to_field(value):
    """
    Parses a pointer info field. Supports both JSON arrays and single dicts.
    Returns the number of parameters and total number of points-to types.
    """
    try:
        if isinstance(value, str):
            value = value.replace("'", '"')
        parsed = json.loads(value)
    except Exception:
        return 0, 0

    if isinstance(parsed, dict):
        pts = parsed.get("points_to", [])
        return 1, len(pts)

    elif isinstance(parsed, list):
        total_pts = 0
        count = 0
        for item in parsed:
            if isinstance(item, dict):
                pts = item.get("points_to", [])
                total_pts += len(pts)
            count += 1  
        return count, total_pts

    return 0, 0

def extract_refined_features(row):
    # Pointer visit
    recv_visit_null = 1 if row['receiver_visit'] in ('null', None, '') else 0
    param_visit_count, pts_visit_total = parse_points_to_field(row['parameters_visit'])

    # Pointer add
    recv_add_null = 1 if row['receiver_add'] in ('null', None, '') else 0
    param_add_count, pts_add_total = parse_points_to_field(row['parameters_add'])

    # Deltas
    delta_params = param_add_count - param_visit_count
    delta_pts = pts_add_total - pts_visit_total

    # Call graph info
    def to_int(x):
        try:
            return int(x)
        except:
            return 0

    visit_out = to_int(row['call_visit_out'])
    visit_in = to_int(row['call_visit_in'])
    visit_nodes = to_int(row['call_visit_total_nodes'])
    visit_edges = to_int(row['call_visit_total_edges'])

    add_out = to_int(row['call_add_out'])
    add_in = to_int(row['call_add_in'])
    add_nodes = to_int(row['call_add_total_nodes'])
    add_edges = to_int(row['call_add_total_edges'])

    # Call graph deltas
    delta_in = add_in - visit_in
    delta_out = add_out - visit_out
    delta_nodes = add_nodes - visit_nodes
    delta_edges = add_edges - visit_edges

    # id of the row
    method = row['src_node']
    offset = row['offset']
    target = row['target_node']

    return [
        method,
        offset,
        target,
        recv_visit_null,
        param_visit_count,
        pts_visit_total,
        recv_add_null,
        param_add_count,
        pts_add_total,
        delta_params,
        delta_pts,
        visit_in,
        visit_out,
        visit_nodes,
        visit_edges,
        add_in,
        add_out,
        add_nodes,
        add_edges,
        delta_in,
        delta_out,
        delta_nodes,
        delta_edges
    ]



def main():
    # Read the CSV file

    for program in os.listdir(var_repr_dir):
        if not program.endswith("_full_info.csv"):
            continue

        # Read the CSV file
        df = pd.read_csv(f"{var_repr_dir}/{program}")

        # Extract features
        features = df.apply(extract_refined_features, axis=1)

        # Create a new DataFrame with the extracted features
        feature_df = pd.DataFrame(features.tolist(), columns=[
            'method', 'offset', 'target',
            'recv_visit_null', 'param_visit_count', 'pts_visit_total',
            'recv_add_null', 'param_add_count', 'pts_add_total',
            'delta_params', 'delta_pts',
            'visit_in', 'visit_out', 'visit_nodes', 'visit_edges',
            'add_in', 'add_out', 'add_nodes', 'add_edges',
            'delta_in', 'delta_out', 'delta_nodes', 'delta_edges'
        ])

        # Save the new DataFrame to a CSV file
        feature_df.to_csv(f"{var_embeddings_dir}/{program}_features.csv", index=False)
    # df = pd.read_csv(f"{var_repr_dir}/urlead5353366_Zeldon_BigData_Class_tgz-pJ8-LocalTestsJ8_full_info.csv")

    # # Extract features
    # features = df.apply(extract_refined_features, axis=1)

    # # Create a new DataFrame with the extracted features
    # feature_df = pd.DataFrame(features.tolist(), columns=[
    #     'method', 'offset', 'target',
    #     'recv_visit_null', 'param_visit_count', 'pts_visit_total',
    #     'recv_add_null', 'param_add_count', 'pts_add_total',
    #     'delta_params', 'delta_pts',
    #     'visit_in', 'visit_out', 'visit_nodes', 'visit_edges',
    #     'add_in', 'add_out', 'add_nodes', 'add_edges',
    #     'delta_in', 'delta_out', 'delta_nodes', 'delta_edges'
    # ])

    # # Save the new DataFrame to a CSV file
    # feature_df.to_csv(f"{var_embeddings_dir}/urlead5353366_Zeldon_BigData_Class_tgz-pJ8-LocalTestsJ8_features.csv", index=False)


if __name__ == "__main__":
    main()