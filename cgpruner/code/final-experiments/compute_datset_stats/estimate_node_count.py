import os
import csv

TRAINING_LIST = "../../../dataset-high-precision-callgraphs/train_programs.txt"
TEST_LIST = "../../../dataset-high-precision-callgraphs/test_programs.txt"
CALLGRAPHS_FOLDER = "../../../dataset-high-precision-callgraphs/full_callgraphs_set"

'''
Compute the list of programs in the datset
'''
dataset = set()
with open(TRAINING_LIST) as f:
    for line in f:
        dataset.add(line.rstrip())
'''
with open(TEST_LIST) as f:
    for line in f:
        dataset.add(line.rstrip())
'''

for benchmark in os.listdir(CALLGRAPHS_FOLDER):
    # only run for the dataset benchmarks
    if benchmark not in dataset:
        continue
    
    wala_file = CALLGRAPHS_FOLDER + "/" + benchmark + "/wala0cfa.csv"
    with open(wala_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        nodes = set()
        edge_count = 0
        for row in csv_reader:
            edge_count += 1
            nodes.add(row["method"])
            nodes.add(row["target"])

    print(f'{benchmark},{len(nodes)},{edge_count}')
