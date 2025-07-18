import os
import pandas as pd

old_ml_dir = '/20TB/mohammad/xcorpus-total-recall/manual_labeling'
new_ml_dir = '/20TB/mohammad/xcorpus-total-recall/manual_labeling2'


def check_and_replace_labels():
    """
    Checks for outdated labels by validating them against the correct
    source file (intersection or disagreement) based on 'source_category'.
    """
    programs = ['axion','batik', 'xerces', 'jasml'] # Add your other programs here
    key_cols = ['method', 'offset', 'target']

    for program in programs:
        print(f"Processing program: {program}")
        old_labeling_path = os.path.join(old_ml_dir, program, 'labeling_sample.csv')
        if not os.path.exists(old_labeling_path):
            print(f"Old labeling file for {program} does not exist. Skipping.")
            continue

        old_labels_df = pd.read_csv(old_labeling_path)
        new_labels_df = old_labels_df.copy()

        # Load the new, updated source data files
        intersection_path = os.path.join(new_ml_dir, program, 'intersection.csv')
        disagreement_path = os.path.join(new_ml_dir, program, 'disagreement.csv')

        if not os.path.exists(intersection_path) or not os.path.exists(disagreement_path):
            print(f"New data files for {program} do not exist. Skipping.")
            continue

        intersection_df = pd.read_csv(intersection_path)
        disagreement_df = pd.read_csv(disagreement_path)

        for index, row in old_labels_df.iterrows():
            # =================================================================
            # KEY LOGIC: Using 'source_category' to direct the check
            # =================================================================
            source_category = row['source_category']
            
            # 1. We determine which new dataframe to use for validation
            #    based on the 'source_category' of the current row.
            if source_category == 'intersection':
                target_df = intersection_df
            elif source_category == 'disagreement':
                target_df = disagreement_df
            else:
                # If source_category is something else, skip this row
                continue

            # 2. Now, we check if the data point (identified by key_cols) 
            #    exists in the *correctly chosen* target dataframe.
            row_identifier = pd.DataFrame([row[key_cols]])
            merged = pd.merge(row_identifier, target_df, on=key_cols, how='inner')

            # 3. If the merge is empty, the data point is not in its
            #    respective new file, so it's outdated and must be replaced.
            if merged.empty:
                print(f"\nOutdated row found in '{program}' from '{source_category}':")
                print(row.to_frame().T.to_string(index=False))

                # Find a new, unused sample from the same category
                candidates = pd.merge(target_df, old_labels_df, on=key_cols, how='left', indicator=True)
                new_candidates = candidates[candidates['_merge'] == 'left_only']

                if not new_candidates.empty:
                    replacement_sample = new_candidates.sample(n=1)
                    new_row_data = {
                        'method': replacement_sample['method'].iloc[0],
                        'offset': replacement_sample['offset'].iloc[0],
                        'target': replacement_sample['target'].iloc[0],
                        'source_category': source_category  # Keep the original category
                    }
                    new_labels_df.loc[index] = new_row_data
                    print("Replaced with a new random sample:")
                    print(pd.DataFrame([new_row_data]).to_string(index=False))
                else:
                    print(f"Warning: No new candidates available for replacement in '{program}' for category '{source_category}'.")

        # Save the updated dataframe to the new directory
        new_labeling_dir = os.path.join(new_ml_dir, program)
        os.makedirs(new_labeling_dir, exist_ok=True)
        new_labeling_path = os.path.join(new_labeling_dir, 'labeling_sample.csv')
        new_labels_df.to_csv(new_labeling_path, index=False)
        print(f"\nUpdated labeling file for '{program}' saved to: {new_labeling_path}")


# Run the function
check_and_replace_labels()