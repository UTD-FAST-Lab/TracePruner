import os
import csv

CALLGRAPHS_FOLDER = "../../../dataset-high-precision-callgraphs/full_callgraphs_set"
RESULT_FOLDER = "wala_callgraphs"

#Loop through the benchmarks
for benchmark in os.listdir(CALLGRAPHS_FOLDER):
    benchmark_callgraph = f'{CALLGRAPHS_FOLDER}/{benchmark}/wala0cfa.csv'

    with open(benchmark_callgraph, "r") as rp:
        csv_reader = csv.reader(rp, delimiter=',')
        with open(RESULT_FOLDER + '/' + benchmark, 'w') as wp:
            csv_writer = csv.writer(wp, delimiter=',')
            for row in csv_reader:
                new_row = [row[29],row[30],row[31],row[1],row[2]]
                csv_writer.writerow(new_row)

