import os
import numpy as np
import csv
from gensim.models import Word2Vec

# 🟢 Step 1: Load Traces from Files
def load_traces(data_folder):
    all_sequences = []
    
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)
        
        with open(filepath, 'r') as f:
            # Convert each line into (int, int) and format as "int_int" string
            sequence = [tuple(map(int, line.strip().split(',')))
                for line in f.readlines()
                if line.strip() and line.lstrip()[0].isdigit() and len(line.split(',')) == 2]
            sequence = [f"{x[0]}_{x[1]}" for x in sequence]  # Convert to Word2Vec-friendly format
            
            all_sequences.append(sequence)
    
    return all_sequences

# 🟢 Step 2: Train Word2Vec Model
def train_word2vec(sequences, vector_size=64, window=5, min_count=1, workers=4):
    model = Word2Vec(sentences=sequences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

# 🟢 Step 3: Encode Traces into 64-Dimensional Vectors
def encode_trace(trace, model, vector_size=64):
    vectors = [model.wv[word] for word in trace if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)  # Return mean vector

def encode_all_traces(traces, model, vector_size=64):
    return [encode_trace(trace, model, vector_size) for trace in traces]

# 🟢 Step 4: Save Encoded Traces to CSV
def save_to_csv(encoded_traces, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Trace_ID"] + [f"Feature_{i}" for i in range(64)])  # Header
        
        for i, vector in enumerate(encoded_traces):
            writer.writerow([f"Trace_{i}"] + list(vector))

# 🔵 Main Execution
if __name__ == "__main__":
    data_folder = "/home/mohammad/projects/CallGraphPruner/data/encoded-edge/url89cafed56a_AlexDiru_esper_compiler_final_tgz-pJ8-compiler_ProgramJ8/edges"  # ✅ Change this to your trace folder
    output_csv = "encoded_traces.csv"

    print("🔍 Loading traces...")
    traces = load_traces(data_folder)
    
    print("🧠 Training Word2Vec model...")
    w2v_model = train_word2vec(traces)
    
    print("📌 Encoding traces into 64D feature vectors...")
    encoded_traces = encode_all_traces(traces, w2v_model)
    
    print("💾 Saving to CSV...")
    save_to_csv(encoded_traces, output_csv)

    print(f"✅ Encoding complete! Saved to {output_csv}")
