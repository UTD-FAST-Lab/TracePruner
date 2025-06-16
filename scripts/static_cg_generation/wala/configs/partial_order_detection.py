import os
import pandas as pd
import json

# === Paths ===
static_cgs_dir = '/20TB/mohammad/xcorpus-total-recall/static_cgs/wala/final'
output_dir = '/20TB/mohammad/xcorpus-total-recall/compare'
json_file = "/home/mohammad/projects/TracePruner/scripts/static_cg_generation/wala/configs/wala_config.json"

# === Config CSVs ===
config_files = [
    "wala_default_1change_configs_v1.csv",
    "wala_1change_configs_v2.csv",
    "wala_1change_configs_v3.csv",
]
config_file_versions = ["v1", "v2", "v3"]  # Must match order

config_dir = "/home/mohammad/projects/TracePruner/scripts/static_cg_generation/wala/configs"


def has_partial_order(orders, analysis1, analysis2):
    for order in orders:
        if order["left"] == analysis2 and order["right"] == analysis1:
            print(f"Found partial order: {analysis2} < {analysis1}")
            return True
    return False


def find_diff(version1, id1, version2, id2):

    for program in os.listdir(static_cgs_dir):
        file1 = os.path.join(static_cgs_dir, program, f"wala_{version1}_{id1}.csv")
        file2 = os.path.join(static_cgs_dir, program, f"wala_{version2}_{id2}.csv")

        if not os.path.exists(file1) or not os.path.exists(file2):
            print(f"Missing: {file1} or {file2}")
            return

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        print("loadded dfs")

        key_cols = ["method", "offset", "target"]
        # diff_df = df1.merge(df2[key_cols], on=key_cols, how='left', indicator=True)
        # df_only_in_df1 = diff_df[diff_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        # Convert rows to sets of tuples for comparison
        df1_keys = set(map(tuple, df1[key_cols].values))
        df2_keys = set(map(tuple, df2[key_cols].values))

        # Compute set difference
        diff_keys = df1_keys - df2_keys

        # Create a DataFrame from diff_keys
        if diff_keys:
            df_only_in_df1 = pd.DataFrame(list(diff_keys), columns=key_cols)
        else:
            df_only_in_df1 = pd.DataFrame(columns=key_cols)

        print("computed diff")
        if not df_only_in_df1.empty:
            print("righting to file")
            diff_dir = os.path.join(output_dir, 'wala', 'final', program)
            os.makedirs(diff_dir, exist_ok=True)
            name = f"{version1}_{id1}__{version2}_{id2}_diff.csv"
            df_only_in_df1.to_csv(os.path.join(diff_dir, name), index=False)


def configs_match_except_analysis(config1, config2):

    for key in config1:
        if key == 'analysis' or key == 'config_id' or key == 'config_version':
            continue
        if config1.get(key) != config2.get(key):
            return False
    return True


def read_all_configs():
    all_configs = []
    for file, version in zip(config_files, config_file_versions):
        path = os.path.join(config_dir, file)
        df = pd.read_csv(path)
        df["config_id"] = df.index
        df["config_version"] = version
        all_configs.extend(df.to_dict(orient="records"))
    return all_configs


def main():
    with open(json_file, 'r') as f:
        orders = next(opt["orders"] for opt in json.load(f)['options'] if opt["name"] == "cgalgo")

    configs = read_all_configs()
    diff_pairs = []

    for i, config in enumerate(configs):
        for j, config_compare in enumerate(configs):
            if i == j:
                continue
            if configs_match_except_analysis(config, config_compare):
                if has_partial_order(orders, config["analysis"], config_compare["analysis"]):
                    diff_pairs.append((config, config_compare))

    print(f"Found {len(diff_pairs)} config pairs with partial order.")

    for c1, c2 in diff_pairs:
        find_diff(c1["config_version"], c1["config_id"], c2["config_version"], c2["config_id"])


if __name__ == "__main__":
    main()
