#!/usr/bin/env python3
'''This script combines all precision, recall, values and so on to 
produce the graphs for 
a) the introduction
b) evaluation (precision-recall)
c) evaluation (fscore-cutoff)
'''

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
values = []

filename = sys.argv[1]
metric = sys.argv[2]
word1 = sys.argv[3]

with open(filename, 'r') as readfp:
    csv_reader = csv.reader(readfp, delimiter=',')
    for row in csv_reader:
        if metric=="Precision":
            values.append(float(row[1]))
        elif metric=="Recall":
            values.append(float(row[2]))

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(10)

labelx = word1 + " " + metric
output_fig = filename + "_" + metric + ".pdf"

large_benchmarks = plt.hist(values, bins=25, histtype='bar', ec='black')
plt.xlabel(labelx, fontsize=16)
plt.ylabel('Number of benchmarks', fontsize=16)
plt.savefig(output_fig,dpi=DPI_VALUE, bbox_inches='tight')

