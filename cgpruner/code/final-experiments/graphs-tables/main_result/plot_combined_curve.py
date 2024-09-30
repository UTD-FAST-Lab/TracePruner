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

wala_prec = []
wala_rec = []

peta_prec = []
peta_rec = []

doop_prec = []
doop_rec = []

rta_prec = []
rta_rec = []

#Read in the arrays
with open("wala.csv", 'r') as readfp:
    csv_reader = csv.reader(readfp, delimiter=',')
    for row in csv_reader:
        wala_prec.append(float(row[1]))
        wala_rec.append(float(row[2]))

with open("doop.csv", 'r') as readfp:
    csv_reader = csv.reader(readfp, delimiter=',')
    for row in csv_reader:
        doop_prec.append(float(row[1]))
        doop_rec.append(float(row[2]))

with open("peta.csv", 'r') as readfp:
    csv_reader = csv.reader(readfp, delimiter=',')
    for row in csv_reader:
        peta_prec.append(float(row[1]))
        peta_rec.append(float(row[2]))

with open("rta.csv", 'r') as readfp:
    csv_reader = csv.reader(readfp, delimiter=',')
    for row in csv_reader:
        rta_prec.append(float(row[1]))
        rta_rec.append(float(row[2]))

['-', '--', '-.', ':']
#Now plot the 3 in a single precision-recall graph.
fig, ax = plt.subplots()


our_doop = ax.plot(doop_prec,doop_rec,color='green', linestyle='--', label='call-graph pruner on Doop')
doop_baseline = ax.scatter([0.2308],[0.9206],color='green', marker = '^', label='Baseline Doop 0CFA')

our_peta = ax.plot(peta_prec,peta_rec,color='blue', linestyle=':', label='call-graph pruner on Petablox')
peta_baseline = ax.scatter([0.2976],[0.8875],color='blue', marker = '^', label='Baseline Petablox 0CFA')

our_wala = ax.plot(wala_prec,wala_rec,color='black', linestyle='-', label='call-graph pruner on Wala')
wala_baseline = ax.scatter([0.2384],[0.9534],color='black', marker = '^', label='Baseline Wala 0CFA')
wala_1cfa = ax.scatter([0.2964],[0.954],color='red', marker = 'P', label='Wala 1CFA')

#our_rta = ax.plot(rta_prec,rta_rec,color='red', linestyle='-.', label='cg-pruner on Wala RTA')
#rta_baseline = ax.scatter([0.1472],[0.9557],color='red', marker = 'P', label='Baseline Wala RTA')

doop_max_fscore = ax.scatter([0.662],[0.662],color='green', marker = 's',label='Doop Equal Precision-Recall point') 
peta_max_fscore = ax.scatter([0.6638],[0.6638],color='blue', marker = 's',label='Petablox Equal Precision-Recall point') 
wala_max_fscore = ax.scatter([0.66],[0.66],color='black', marker = 's',label='Wala Equal Precision-Recall point')
#rta_max_fscore = ax.scatter([0.6858],[0.6858],color='red', marker = 's') 

doop_intro_point = ax.scatter([0.50],[0.88],color='green', marker = 'o')
peta_intro_point = ax.scatter([0.50],[0.87],color='blue', marker = 'o')
wala_intro_point = ax.scatter([0.50],[0.92],color='black', marker = 'o')
#rta_intro_point = ax.scatter([0.3004],[0.9493],color='red', marker = '*')

wala_generic_classifier = ax.scatter([0.4914],[0.9243],color='orange', marker = 'x',label='Generic Classifier for Wala')

#Add the limits, legend and axis names
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
#plt.gca().set_aspect('equal', adjustable='box')
legend= plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),ncol=2)
plt.savefig('main-result.pdf',bbox_extra_artists=(legend,), bbox_inches='tight',dpi=DPI_VALUE)

