import os
import pandas as pd

static_cgs_dir = '/20TB/mohammad/xcorpus-total-recall/static_cgs/doop/final'

for program in os.listdir(static_cgs_dir):
    program_dir = os.path.join(static_cgs_dir, program)
    if not os.path.isdir(program_dir):
        continue  # skip if not a directory

    for scg_file in os.listdir(program_dir):
        if scg_file.endswith('.csv'):
            scg_path = os.path.join(program_dir, scg_file)

            try:
                df = pd.read_csv(scg_path, dtype=str)
            except Exception as e:
                print(f"Failed to read {scg_path}: {e}")
                continue

            # Remove rows with non-integer offsets
            valid_offset_mask = pd.to_numeric(df['offset'], errors='coerce').notna()
            invalid_rows = df[~valid_offset_mask]

            if not invalid_rows.empty:
                print(f"Invalid offsets found in {scg_path}:")
                print(invalid_rows[['offset']].drop_duplicates())

            # Keep only rows with valid integer offsets
            df = df[valid_offset_mask]

            # (Optional) Convert offset to int if needed
            # df['offset'] = df['offset'].astype(int)

            # (Optional) Save cleaned version back (overwrite or new file)
            df.to_csv(scg_path, index=False)