import os
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from utils.utils import load_code, get_input_and_mask, read_config_file
from utils.converter import convert

class CallGraphFinetuneDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.raw_data_path = self.config["BENCHMARK_CALLGRAPHS"]
        self.processed_path = self.config["PROCESSED_DATA"]
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.max_length = 512

        if self.mode == "train":
            self.program_lists = self.config["TRAINING_PROGRAMS_LIST"]
        elif self.mode == "test":
            self.program_lists = self.config["TEST_PROGRAMS_LIST"]
        else:
            raise NotImplementedError("Invalid mode: {}".format(self.mode))

        self.data = []
        self.labels = []

        # Load data
        with open(self.program_lists, "r") as f:
            for line in f:
                program_name = line.strip()
                wala_path = os.path.join(self.raw_data_path, program_name, "wala0cfa.csv")
                code_path = os.path.join(self.processed_path, program_name, "code.csv")

                if not os.path.exists(wala_path) or not os.path.exists(code_path):
                    print(f"Skipping {program_name} due to missing files.")
                    continue

                df = pd.read_csv(wala_path)
                descriptor2code = load_code(code_path)

                for i in range(len(df)):
                    method = df.loc[i, "method"]
                    target = df.loc[i, "target"]
                    label = df.loc[i, "label"]
                    if label == -1:
                        continue

                    # Skip <boot> edges
                    if method == "<boot>":
                        continue

                    # Resolve source and target code
                    method_code = descriptor2code.get(method, convert(method).__tocode__())
                    target_code = descriptor2code.get(target, convert(target).__tocode__())

                    # Tokenize and mask
                    token_ids, mask = get_input_and_mask(method_code, target_code, self.max_length, self.tokenizer)

                    # Store data
                    self.data.append((token_ids, mask))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids, mask = self.data[idx]
        label = self.labels[idx]
        return {
            "ids": torch.tensor(token_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }
