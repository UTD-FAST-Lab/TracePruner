import sys
import os
#from statistics import geometric_mean
import statistics

TEST_PROGRAMS_LIST = "../../../dataset-high-precision-callgraphs/test_programs.txt"
ORIGINAL_TIMINGS = "../../../dataset-high-precision-callgraphs/overhead_calculation_data/original_timings"
CGPRUNER_TIMINGS = "../../../dataset-high-precision-callgraphs/overhead_calculation_data/1cfa_timings"

with open(TEST_PROGRAMS_LIST) as rp:
    test_programs = [line.rstrip() for line in rp]

overheads = []
for program in test_programs:
    with open(ORIGINAL_TIMINGS + "/" + program + ".txt") as rp2:
        original_time = float(rp2.read())

    with open(CGPRUNER_TIMINGS + "/" + program + ".txt") as rp3:
        cgpruner_time = float(rp3.read())

    overhead = cgpruner_time / original_time
    '''
    if overhead > 2.0:
        continue
    '''    
    #print(overhead)
    overheads.append(overhead)

print(statistics.mean(overheads))
print(statistics.stdev(overheads))
