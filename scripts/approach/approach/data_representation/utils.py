
import os
import pandas as pd

def load_finetuned_semantic_features(semantic_features_dir):
    test_df = pd.read_csv(os.path.join(semantic_features_dir, 'enriched_ft_test.csv'))
    train_df = pd.read_csv(os.path.join(semantic_features_dir, 'enriched_ft_train.csv'))

    combined_df = pd.concat([test_df, train_df], ignore_index=True)

    semantic_map = {}

    for _, row in combined_df.iterrows():
        key = (
            row['program_name'],
            row['method'],
            row['offset'],
            row['target']
        )
        try:
            code_str = row['code']
            code_vec = [float(x) for x in code_str.strip().split(',')]
            semantic_map[key] = code_vec
        except Exception:
            print(f"Error processing row: {row}")

    return semantic_map