import os
import pandas as pd

# Configuration
# root_dir = "/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9"
root_dir = "/home/mohammad/projects/CallGraphPruner/data/static-cgs/static_cgs_no_exlusion"
output_csv = "combined_diff_0cfa_1obj_v2.csv"

# Collect all dataframes
all_dfs = []

for program in os.listdir(root_dir):
    false_df_path = os.path.join(root_dir, program, "diff_0cfa_1obj.csv")

    if not os.path.exists(false_df_path):
        continue

    # Read the CSV
    df = pd.read_csv(false_df_path)

    # Add the program column as the first column
    df.insert(0, "program", program)

    # Append to the list
    all_dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(all_dfs, ignore_index=True)

# Save to a single CSV file
combined_df.to_csv(output_csv, index=False)

print(f"✅ Combined data saved to {output_csv} with {len(combined_df)} rows.")
