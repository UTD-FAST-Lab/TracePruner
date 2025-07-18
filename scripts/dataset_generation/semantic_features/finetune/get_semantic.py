import os
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.utils import load_code, get_input_and_mask
from utils.converter import convert
from finetune_model import BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

model = BERT()
model.load_state_dict(torch.load("finetuned_codebert.pth"))
model.to(device)
model.eval()

# === CONFIG ===
BENCHMARK_CALLGRAPHS = "/20TB/mohammad/my-dataset/no_libs"      # contains <program>/wala0cfa.csv
PROCESSED_DATA = "/20TB/mohammad/baselines/autopruner/replication_package/processed_data/wala"      # contains <program>/code.csv
PROGRAM_LIST_FILE = "/20TB/mohammad/data/programs.txt"    # or training list

OUTPUT_DIR = "/20TB/mohammad/data/semantic_finetuned"                  # where program-wise CSVs go

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load program names ===
with open(PROGRAM_LIST_FILE, "r") as f:
    program_list = [line.strip() for line in f.readlines()]

# === Process each program ===
for program in program_list:
    print(f"Processing {program}...")
    wala_path = os.path.join(BENCHMARK_CALLGRAPHS, program, "wala0cfa.csv")
    code_path = os.path.join(PROCESSED_DATA, program, "code.csv")
    output_csv = os.path.join(OUTPUT_DIR, f"{program}.csv")

    if not os.path.exists(wala_path) or not os.path.exists(code_path):
        print(f"Skipping {program} due to missing files.")
        continue

    df = pd.read_csv(wala_path)
    descriptor2code = load_code(code_path)
    rows = []

    for i in tqdm(range(len(df))):
        method = df.loc[i, "method"]
        offset = df.loc[i, "offset"]
        target = df.loc[i, "target"]

        # Skip <boot> edges
        if method == "<boot>":
            continue

        # Resolve source and target code
        if method in descriptor2code:
            method_code = descriptor2code[method]
        else:
            method_code = convert(method).__tocode__()

        if target in descriptor2code:
            target_code = descriptor2code[target]
        else:
            target_code = convert(target).__tocode__()

        # Tokenize as in get_input_and_mask()
        token_ids, mask = get_input_and_mask(method_code, target_code, 512, tokenizer)
        ids_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
        mask_tensor = torch.tensor(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            _, embedding = model(ids_tensor, mask=mask_tensor)
        
        embedding_np = embedding.squeeze(0).cpu().numpy()
        row = [method, offset, target] + embedding_np.tolist()
        rows.append(row)

    # Save to CSV
    columns = ["method", "offset", "target"] + [f"emb_{i}" for i in range(768)]
    pd.DataFrame(rows, columns=columns).to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
