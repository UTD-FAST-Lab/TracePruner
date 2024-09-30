#!/usr/bin/env python3
'''This script combines all precision, recall, values and so on to 
produce the graphs for 
a) the introduction
b) evaluation (precision-recall)
c) evaluation (fscore-cutoff)
'''
import numpy as np
import sys
import pathlib
import csv
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DPI_VALUE = 600
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

original_filename = sys.argv[1]
pruned_filename = sys.argv[2]

old_values = {}
improvement = []

with open(original_filename, 'r') as readfp:
    csv_reader = csv.reader(readfp, delimiter=',')
    for row in csv_reader:
        old_values[row[0]] = float(row[1])

with open(pruned_filename, 'r') as readfp:
    csv_reader = csv.reader(readfp, delimiter=',')
    for row in csv_reader:
        benchmark = row[0].split("/")[-1]
        old_precision = old_values[benchmark]
        new_precision = float(row[1])
        imp = (new_precision - old_precision)*100/old_precision
        if imp == 0:
            continue
        improvement.append(imp)

fig, ax = plt.subplots()
fig.set_figheight(3.5)
fig.set_figwidth(10)

ax.set_xscale('log')
num_bins = np.geomspace(1, max(improvement), 25)

large_benchmarks = plt.hist(improvement, bins=num_bins, histtype='bar', ec='black')
plt.xlabel("% Improvement in Precision score", fontsize=16)
plt.ylabel('Number of benchmarks', fontsize=16)
plt.savefig('precision_improvement.pdf',dpi=DPI_VALUE, bbox_inches='tight')

# print it
improvement.sort()
for i in improvement:
    print(i)