import os
import pandas as pd

manual_dir = "/20TB/mohammad/xcorpus-total-recall/manual_labeling/overall"

def main():
    
    total_unknowns = []
    for program in os.listdir(manual_dir):
        program_dir = os.path.join(manual_dir, program)
        if not os.path.isdir(program_dir):
            continue
        
        unk_file = pd.read_csv(os.path.join(program_dir, f'unknown_{program}.csv'))

        total_unknowns.append(unk_file)

    
    # randomly sample 350 from the total unknowns and put into batches of 50
    if total_unknowns:
        total_unknowns = pd.concat(total_unknowns).drop_duplicates(subset=['method', 'offset', 'target'])
        # shoufle and sample 350 unknown edges
        total_unknowns = total_unknowns.sample(frac=1, random_state=42).reset_index(drop=True)
        if len(total_unknowns) > 350:
            # If there are more than 350 unknown edges, sample 350
            print(f"Sampling 350 unknown edges from {len(total_unknowns)} total unknown edges.")

        total_unknowns = total_unknowns.sample(n=350, random_state=42)
        
        for i in range(0, len(total_unknowns), 50):
            batch = total_unknowns.iloc[i:i+50]
            batch_path = os.path.join(manual_dir, 'batches', f'batch_{i//50 + 1}.csv')
            os.makedirs(os.path.dirname(batch_path), exist_ok=True)
            batch.to_csv(batch_path, index=False)
            print(f"Saved batch {i//50 + 1} with {len(batch)} unknown edges to {batch_path}")
   


if __name__ == "__main__":
    main()