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

rta_prec = []
rta_rec = []

with open("rta.csv", 'r') as readfp:
    csv_reader = csv.reader(readfp, delimiter=',')
    for row in csv_reader:
        rta_prec.append(float(row[1]))
        rta_rec.append(float(row[2]))

fig, ax = plt.subplots()
our_rta = ax.plot(rta_prec,rta_rec,color='red', linestyle='-.', label='cg-pruner on Wala RTA')
rta_baseline = ax.scatter([0.1472],[0.9557],color='red', marker = 'P', label='Baseline Wala RTA')
rta_max_fscore = ax.scatter([0.6678],[0.6678],color='red', marker = 's') 

      
#Add the limits, legend and axis names
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.gca().set_aspect('equal', adjustable='box')
legend= plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),ncol=2)
plt.savefig('rta.pdf',bbox_extra_artists=(legend,), bbox_inches='tight',dpi=DPI_VALUE)

