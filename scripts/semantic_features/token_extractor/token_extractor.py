import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, T5EncoderModel,AutoTokenizer
from utils.utils import load_code, get_input_and_mask
from utils.converter import convert

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TOKENIZER ===
# tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small', use_fast=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# === CONFIG ===
BENCHMARK_CALLGRAPHS = "/20TB/mohammad/xcorpus-total-recall/features/struct"
PROCESSED_DATA = "/20TB/mohammad/xcorpus-total-recall/features/semantic/source_code"
OUTPUT_DIR = "/20TB/mohammad/xcorpus-total-recall/features/semantic/tokens"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === PROGRAMS ===
program_list = ['axion', 'batik', 'jasml', 'xerces']

# === PROCESS EACH PROGRAM ===
for program in program_list:
    print(f"\nProcessing {program}...")
    program_dir = os.path.join(BENCHMARK_CALLGRAPHS, program)
    for scg_file in os.listdir(program_dir):
        print(f"processing {scg_file}")
        wala_path = os.path.join(program_dir, scg_file)
        code_path = os.path.join(PROCESSED_DATA, program, "code.csv")
        scg_file = scg_file.replace(".csv", ".npz")
        scg_file = scg_file.replace("struct", "token")
        output_npz = os.path.join(OUTPUT_DIR, program, scg_file)
        os.makedirs(os.path.dirname(output_npz), exist_ok=True)

        if not os.path.exists(wala_path) or not os.path.exists(code_path):
            print(f"Skipping {program} due to missing files.")
            continue

        df = pd.read_csv(wala_path)
        descriptor2code = load_code(code_path)
        rows = []
        token_cache = {}  # key: (method, target), value: (token_ids, mask)

        for i in tqdm(range(len(df))):
            method = df.loc[i, "method"]
            offset = df.loc[i, "offset"]
            target = df.loc[i, "target"]

            if method == "<boot>":
                continue

            cache_key = (method, target)
            if cache_key in token_cache:
                token_ids, mask = token_cache[cache_key]
            else:
                # Resolve source code
                method_code = descriptor2code.get(method) or convert(method).__tocode__()
                target_code = descriptor2code.get(target) or convert(target).__tocode__()

                token_ids, mask = get_input_and_mask(method_code, target_code, 512, tokenizer)
                token_cache[cache_key] = (token_ids, mask)

            rows.append({
                "method": method,
                "offset": offset,
                "target": target,
                "tokens": token_ids,
                "masks": mask,
            })

        np.savez_compressed(output_npz, data=rows)
        print(f"Saved: {output_npz}")
