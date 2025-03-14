import pandas as pd

# 1. Load both CSV files
df1 = pd.read_csv("/home/mohammad/projects/CallGraphPruner/data/datasets/branches/combined_w2v_dataset_v0.csv")
df2 = pd.read_csv("/home/mohammad/projects/CallGraphPruner/data/datasets/cgs/combined_w2v_dataset_v0.csv")

# 2. (Optional) Validate that the files have the same number of rows
if len(df1) != len(df2):
    raise ValueError("The two files have different numbers of rows!")

# 3. (Optional) Check that key columns match row by row
#    This ensures row i in df1 corresponds to row i in df2.
#    For example, check edge_name, program_name, label:
if not (df1["edge_name"].equals(df2["edge_name"]) and
        df1["program_name"].equals(df2["program_name"]) and
        df1["label"].equals(df2["label"])):
    raise ValueError("Mismatch between rows in file1 and file2!")

# 4. Identify the feature columns in df2 to add
#    (Exclude columns that appear in df1, like edge_name/program_name/label)
exclude_cols = ["edge_name", "program_name", "label",'method','target']
df2_features = [col for col in df2.columns if col not in exclude_cols]

# 5. Concatenate horizontally
#    This creates a single DataFrame with df1’s columns + df2’s features
df_combined = pd.concat([df1, df2[df2_features]], axis=1)

df_combined = df_combined.drop_duplicates(subset=['program_name', 'method', 'target'], keep='last')

# 6. Write to a new CSV
df_combined.to_csv("combined_128_features.csv", index=False)
