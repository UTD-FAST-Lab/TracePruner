import os
import numpy as np
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# Define LSTM model
model = Sequential([
    Input(shape=(None, 2)),  # Explicitly define input shape
    LSTM(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(8, activation='relu')
])

# Directory paths
input_dir = "/home/mohammad/projects/CallGraphPruner/data/encoded-edge/url72c32f3c54_Quickhull3d_quickhull3d_tgz-pJ8-com_github_quickhull3d_SimpleExampleJ8/edges"
output_csv = "/home/mohammad/projects/CallGraphPruner/data/encoded-edge/url72c32f3c54_Quickhull3d_quickhull3d_tgz-pJ8-com_github_quickhull3d_SimpleExampleJ8/encoded_features.csv"

# Open CSV file for writing
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header row (optional)
    writer.writerow(["Filename"] + [f"Feature_{i}" for i in range(16)])  # Assuming 16 features from Dense layer

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)

        # Read sequence from file
        with open(filepath, "r") as f:

            lines = f.readlines()  # Read all lines at once

        # Process lines: Strip, filter invalid entries, and convert to (int, int)
            sequence =  [
                tuple(map(int, line.strip().split(',')))
                for line in lines
                if line.strip() and line.lstrip()[0].isdigit() and len(line.split(',')) == 2
            ]
            # sequence = [tuple(map(int, line.strip().split())) for line in f.readlines()]  # Convert to list of (int, int)

            # Convert to NumPy array with shape (1, sequence_length, 2)
            X = np.array(sequence).reshape(1, len(sequence), 2)

            # Get feature vector
            feature_vector = model.predict(X).flatten()  # Flatten to 1D

            # Write to CSV
            writer.writerow([filename] + feature_vector.tolist())

print(f"Feature vectors saved to {output_csv}")
