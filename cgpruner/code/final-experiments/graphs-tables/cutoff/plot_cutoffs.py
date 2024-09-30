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

cutoff = []
prec = []
rec = []
fscore = []

tool = sys.argv[1]

#Read in the arrays
with open(tool + ".csv", 'r') as readfp:
    csv_reader = csv.reader(readfp, delimiter=',')
    for row in csv_reader:
        cutoff.append(float(row[0]))
        prec.append(float(row[1]))
        rec.append(float(row[2]))
        fscore.append(float(row[3]))

#Now plot the 3 in a single precision-recall graph.
fig, ax = plt.subplots()

prec_plot = ax.plot(cutoff,prec,color='black', linestyle='-', label='Precision')
rec_plot = ax.plot(cutoff,rec,color='blue', linestyle=':', label='Recall')
fscore_plot = ax.plot(cutoff,fscore,color='red', linestyle='--', label='F-Score')

#Add the limits, legend and axis names
plt.xlabel('Cutoff')
plt.ylabel('Ratio')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.gca().set_aspect('equal', adjustable='box')
legend= plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),ncol=2)
plt.savefig(tool + '_cutoff.pdf',bbox_extra_artists=(legend,), bbox_inches='tight',dpi=DPI_VALUE)

