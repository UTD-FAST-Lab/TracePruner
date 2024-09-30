#!/usr/bin/env python3
'''This script subsamples the 
lines in "combinationWithExtraFeatures.csv"
'''

import sys
import pathlib
import csv
import shutil
import random

DATASET_FILE = "wala_1cfa.csv"
OUTPUT_FILE = "wala_1cfaSubsampled.csv"
benchmarks_folder = pathlib.Path(sys.argv[1])
SAMPLE_SIZE = 20000

def main():
    #Loop through all the file names and read the rows into joint_dataset
    for testcase in benchmarks_folder.iterdir():
        if not testcase.is_dir(): #skip non-directories
            continue

        #If the output is already present, skip
        if (testcase / OUTPUT_FILE).is_file():
            print("Testcase: " + testcase.name + " - output already exists")
            continue 

        print("Testcase: " + testcase.name)
        line_count = 0

        #Get a line count
        with open(testcase / DATASET_FILE, 'r') as readfp:
            csv_reader = csv.reader(readfp)
            for row in csv_reader:
                line_count += 1

        #If line-count < 20000, simply write the file back
        if (line_count < 20000):
            shutil.copyfile(testcase / DATASET_FILE, testcase / OUTPUT_FILE)

        #Else randomly sample approx 20k lines
        with open(testcase / DATASET_FILE, 'r') as readfp:
            csv_reader = csv.reader(readfp)
            header_line = next(csv_reader)
            with open(testcase / OUTPUT_FILE, mode='w') as write_fp:
                csv_writer = csv.writer(write_fp)
                csv_writer.writerow(header_line)
                for row in csv_reader:
                    #This will approx sample 'SAMPLE SIZE' many lines
                    if random.randint(1,line_count) < SAMPLE_SIZE: 
                        csv_writer.writerow(row)
            
                
if __name__ == "__main__":
    main()
