import sys
import pathlib
import csv
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

'''
DEFINING CONSTANTS
'''
DPI_VALUE = 600

main_label = {
	"wala":'cg-pruner on WALA',
	"peta":'cg-pruner on Petablox',
	"doop":'cg-pruner on Doop'
	}
baseline_label = {
	"wala":'Baseline Wala 0CFA',
	"doop":'Baseline Doop 0CFA',
	"peta":'Baseline Petablox 0CFA'
	} 
baseline_rec = {
	"wala":0.9534,
	"doop":0.9206,
	"peta":0.8875
	} 
baseline_prec = {
	"wala":0.2384,
	"doop":0.2308,
	"peta":0.2976
	}
half_rec = {
	"wala":0.9243,
	"doop":0.8956,
	"peta":0.867
	} 
half_prec = {
	"wala":0.4914,
	"doop":0.4668,
	"peta":0.4892
	}
full_rec = {
	"wala":0.8723,
	"doop":0.6914,
	"peta":0.6646
	} 
full_prec = {
	"wala":0.538,
	"doop":0.5545,
	"peta":0.5696
	}
balanced_rec = {
	"wala":0.66,
	"doop":0.662,
	"peta":0.6638
	} 
balanced_prec = {
	"wala":0.66,
	"doop":0.662,
	"peta":0.6638
	}
prec = []
rec = []

'''
MAIN CODE
'''
tool = sys.argv[1]

#Read in the arrays
with open(tool + ".csv", 'r') as readfp:
    csv_reader = csv.reader(readfp, delimiter=',')
    for row in csv_reader:
        prec.append(float(row[1]))
        rec.append(float(row[2]))

#Now plot in a single precision-recall graph.
fig, ax = plt.subplots()

our_tool = ax.plot(prec,rec,color='black',label=main_label[tool])
baseline = ax.scatter([baseline_prec[tool]],[baseline_rec[tool]],color='black', marker = 'x', label=baseline_label[tool])
half_heuristic = ax.scatter([half_prec[tool]],[half_rec[tool]],color='green', marker = '^', label='Remove edges using the first heuristic-filter rule')
#full_heuristic = ax.scatter([full_prec[tool]],[full_prec[tool]],color='red', marker = '*', label='Remove edges using both heuristic-filter rules')
balanced_point = ax.scatter([balanced_prec[tool]],[balanced_rec[tool]],color='black', marker = 's',label='Pruned call-graph at the Equal Precision-Recall point')
if (tool=="wala"):
	one_cfa = ax.scatter([0.2964],[0.954],color='blue', marker = '.', label='WALA 1CFA')

#Add the limits, legend and axis names
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.05])
plt.gca().set_aspect('equal', adjustable='box')
legend= plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12))
plt.savefig(tool + '.pdf',bbox_extra_artists=(legend,), bbox_inches='tight',dpi=DPI_VALUE)

