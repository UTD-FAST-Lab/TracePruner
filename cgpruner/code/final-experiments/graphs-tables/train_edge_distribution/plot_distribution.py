import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DPI_VALUE = 600
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
values = []

filename = sys.argv[1]

with open(filename, 'r') as readfp:
    csv_reader = csv.reader(readfp, delimiter=',')
    for row in csv_reader:
        values.append(float(row[1]))
            
fig, ax = plt.subplots()
fig.set_figheight(3.5)
fig.set_figwidth(10)

ax.set_xscale('log')
num_bins = np.geomspace(min(values), max(values), 25)

ax.hist(values, bins=num_bins, histtype='bar', ec='black')
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xlabel("Number of Edges", fontsize=16)
plt.ylabel('Number of benchmarks', fontsize=16)
plt.savefig("train_edge_distribution.pdf",dpi=DPI_VALUE, bbox_inches='tight')

