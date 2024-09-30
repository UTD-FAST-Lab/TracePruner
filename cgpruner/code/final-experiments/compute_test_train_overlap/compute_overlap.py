'''
This script computes the main result of the paper
'''

import sys
import csv
import pathlib

sys.path.insert(1, '../../common')
from constants import *


'''
READING CONFIGS, HEADERS AND THE MODEL
'''

# Read command line args
BENCHMARK_CALLGRAPHS = pathlib.Path(sys.argv[1])
TRAIN_PROGRAMS_LIST = sys.argv[2]
TEST_PROGRAMS_LIST = sys.argv[3]
CONFIG_FILE = sys.argv[4]

configs = read_config_file(CONFIG_FILE)
FULL_FILE = configs["FULL_FILE"]

'''
READING TEST DATA
'''
train_edges = set()
train_nodes = set()

with open(TRAIN_PROGRAMS_LIST) as f:
    train_programs = [line.rstrip() for line in f]

for prog in BENCHMARK_CALLGRAPHS.iterdir():
    if prog.stem not in train_programs:
        continue
    train_file = prog / FULL_FILE
    with open(train_file, 'r') as trainfp:
        csv_reader = csv.DictReader(trainfp)
        for row in csv_reader:
            src_method = row[SRC_NODE_COL_NAME]
            dest_method = row[DEST_NODE_COL_NAME]
            train_nodes.add(src_method)
            train_nodes.add(dest_method)
            signature = f'{src_method}#{dest_method}'
            train_edges.add(signature)

#Read the test programs list
with open(TEST_PROGRAMS_LIST) as f:
    test_programs = [line.rstrip() for line in f]

for prog in BENCHMARK_CALLGRAPHS.iterdir():
    if prog.stem not in test_programs:
        continue
    test_file = prog / FULL_FILE
    with open(test_file, 'r') as testfp:
        csv_reader = csv.DictReader(testfp)
        edges = set()
        nodes = set()
        for row in csv_reader:
            src_method = row[SRC_NODE_COL_NAME]
            dest_method = row[DEST_NODE_COL_NAME]
            nodes.add(src_method)
            nodes.add(dest_method)
            signature = f'{src_method}#{dest_method}'
            edges.add(signature)

        node_matches = 0.0
        node_total = 0.0
        for n in nodes:
            node_total += 1
            if n in train_nodes:
                node_matches += 1
        
        edge_matches = 0.0
        edge_total = 0.0
        for e in edges:
            edge_total += 1
            if e in train_edges:
                edge_matches += 1

        node_overlap = node_matches / node_total
        edge_overlap = edge_matches / edge_total
        print(f'{prog.stem},{node_overlap},{edge_overlap}')
