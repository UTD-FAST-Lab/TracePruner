from collections import Counter, defaultdict
import re

# Read the file
file_path = "../data/diff-sim/diff-mt3vc1.txt"

# Storage for tuples and sequences
deleted_tuples = Counter()
deleted_sequences = Counter()
deleted_indexes = []

# Read and process the file
with open(file_path, "r") as file:
    all_tuples = []
    
    for line in file:
        match = re.search(r"Delete: s1\[(\d+)\]=\((\d+), (\d+)\)", line)
        if match:
            index = int(match.group(1))
            tuple_val = (int(match.group(2)), int(match.group(3)))

            # Store all deleted tuples in order
            all_tuples.append(tuple_val)

            # Track individual tuple deletions
            deleted_tuples[tuple_val] += 1

            # Track deleted indexes for range analysis
            deleted_indexes.append(index)

# 1. Most deleted tuples
most_deleted_tuples = deleted_tuples.most_common(10)

# 2. Find frequently appearing sequences of tuples
sequence_counts = defaultdict(int)
sequence_length = 3  # Define the length of sequences to track

for i in range(len(all_tuples) - sequence_length + 1):
    seq = tuple(all_tuples[i : i + sequence_length])  # Extract sequences of 'sequence_length'
    sequence_counts[seq] += 1

# Get the most frequently deleted sequences
most_common_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:10]

# 3. Find index ranges with the most deletions
deleted_indexes.sort()
range_counter = Counter((i // 1000) * 1000 for i in deleted_indexes)  # Grouping by 1000s

most_changed_ranges = range_counter.most_common(20)

# 4. Summary statistics
total_deletions = sum(deleted_tuples.values())
unique_tuples = len(deleted_tuples)

# Print results
print("Most Deleted Tuples:")
for tpl, count in most_deleted_tuples:
    print(f"Tuple {tpl} was deleted {count} times")

print("\nMost Repeated Sequences of Tuples:")
for seq, count in most_common_sequences:
    print(f"Sequence {seq} appeared {count} times")

print("\nMost Changed Index Ranges:")
for rng, count in most_changed_ranges:
    print(f"Range {rng}-{rng+999} has {count} deletions")

print(f"\nTotal unique tuples deleted: {unique_tuples}")
print(f"Total deletions: {total_deletions}")
