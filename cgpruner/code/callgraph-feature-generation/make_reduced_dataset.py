#!/usr/bin/env python3
'''This script generates a 'combinationWithExtraFeatures.csv'
version for just a subset of the tools
'''

import sys
import pathlib
import csv
import shutil
import random

DATASET_FILE = "combinationWithExtraFeatures.csv"
OUTPUT_FILE = "wala_1cfa.csv"
benchmarks_folder = pathlib.Path(sys.argv[1])
ANALYSES_TO_KEEP = [
    "wala-cge-1cfa-noreflect-intf-direct",
    "wala-cge-1cfa-noreflect-intf-trans",
    "wiretap"
]
EXTRA_FEATURES = ["method","offset","target"]
#DYNAMIC_ANALYSIS = "wiretap"
FEATURES_TO_REMOVE = ["graph_num_orphan_nodes",
      "graph_rel_node_count",
      "graph_rel_edge_count",
      "num_paths_to_this_from_main",
      "depth_from_orphans",
]

def main():
    #Loop through all the file names and read the rows into joint_dataset
    for testcase in benchmarks_folder.iterdir():
        if not testcase.is_dir(): #skip non-directories
            continue

        #If the output is already present, skip
        if (testcase / OUTPUT_FILE).is_file():
            print("Testcase: " + testcase.name + " - output already exists")
            continue 
        else:
            print("Testcase: " + testcase.name)
        
        #Actually reading the file
        with open(testcase / DATASET_FILE, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            #compute the correct set of headers
            ignored_cols = set()
            for feat in FEATURES_TO_REMOVE:
                for header in csv_reader.fieldnames:
                    if feat in header:
                        ignored_cols.add(header)
            headers = []
            #Add headers from ANALYSES_TO_KEEP which are not in ignored_cols
            for f in csv_reader.fieldnames:
                if f not in ignored_cols:
                    for analysis in ANALYSES_TO_KEEP:
                        #can't check for name match because all extra-feature
                        #columns also have to be added. Hence using 'in'
                        if analysis in f:
                            headers.append(f)
                            break
            headers += EXTRA_FEATURES

            #Open the file for writing
            with open(testcase / OUTPUT_FILE, "w") as writefp:
                csv_writer = csv.DictWriter(
                    writefp, fieldnames=headers, extrasaction="ignore")
                csv_writer.writeheader()

                for row in csv_reader:
                    #Don't bother keeping rows where they all say 0
                    for analysis in ANALYSES_TO_KEEP:
                        if row[analysis] == "1": 
                            csv_writer.writerow(row)
                            break

if __name__ == "__main__":
    main()
