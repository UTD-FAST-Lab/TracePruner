import sys
import csv
import copy
import random
import pathlib

'''
CONSTANTS
'''
PRINT_PER_PROG_VALUES = True
PROGRAM_LEVEL_EVAL = True
LABEL_COLUMN = "wiretap"
BENCHMARK_ID_COL = "benchmark_id"
POSITIVE_LABEL = 1
SRC_NODE_COL_NAME = "method"
DEST_NODE_COL_NAME = "target"
BC_OFFSET_COL_NAME = "offset"
SA_TRANS_INDEX = 1
SA_DIRECT_INDEX = 0
FEATURE_GROUPS_TO_REMOVE = [
        "#edge_disjoint_paths_from_main",
        "#node_disjoint_paths_from_main"
    ]

SUBSAMPLE_FILE = ""
FULL_FILE = ""
sa_trans_header = ""
sa_direct_header = ""

def read_config_file(config_file):
    configs = {}
    with open(config_file) as f:
        for line in f.readlines():
            line = line.strip()
            split_line = line.split("=")
            configs[split_line[0]] = split_line[1]
    return configs

def compute_headers(benchmarks_folder, subsample_file):
    #Computing the correct headers
    ignored_cols = {"method", "offset", "target", LABEL_COLUMN}
    header_names = []
    for training_prog in benchmarks_folder.iterdir():
        first_training_file = training_prog / subsample_file
        with open(first_training_file, 'r') as readfp:
            csv_reader = csv.DictReader(readfp)
            #Add the FEATURE_GROUPS_TO_REMOVE, to the ignored_cols
            for feat in FEATURE_GROUPS_TO_REMOVE:
                for header in csv_reader.fieldnames:
                    if feat in header:
                        ignored_cols.add(header)
            #Get the correct set of header names
            header_names = [f for f in csv_reader.fieldnames if f not in ignored_cols]
            break
    return header_names

def sanity_check_names(header_names, sa_trans_header, sa_direct_header):
    #sanity check for the indices
    if (SA_TRANS_INDEX != header_names.index(sa_trans_header)
        or SA_DIRECT_INDEX != header_names.index(sa_direct_header)):
      print("ERROR. SA_TRANS_INDEX or SA_DIRECT_INDEX doesnt match")
      print(SA_TRANS_INDEX,header_names.index(sa_trans_header))
      print(SA_DIRECT_INDEX,header_names.index(sa_direct_header))
      sys.exit()

    return header_names
