import numpy as np
import sys
import csv
import copy
import random
import pathlib
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

INPUT_FILE = sys.argv[1]
LEARNED_CLASSIFIER = sys.argv[2]
CUTOFF = float(sys.argv[3])

#Initialization
edge_features = []
edge_names = []

#Reading in the input file
ignored_cols = {"method", "offset", "target"}
with open(INPUT_FILE, 'r') as readfp:
    csv_reader = csv.DictReader(readfp)
    #Get the correct set of header names
    header_names = [f for f in csv_reader.fieldnames if f not in ignored_cols]
    #loop through the data
    for row in csv_reader:
        next_row = []
        for header in header_names:
            next_row.append(float(row[header]))
        edge_features.append(next_row)
        edge_names.append(f'{row["method"]},{row["offset"]},{row["target"]}')

#Convert into array format
edge_features = np.array(edge_features)

#Prediction
clf = load(LEARNED_CLASSIFIER) 
y_pred_proba = clf.predict_proba(edge_features)
y_pred_proba = y_pred_proba[:,1] #just use probability of label 1

#Print the set of edges with probability >= CUTOFF
print("method,offset,target")
for i in range(len(y_pred_proba)):
    if y_pred_proba[i] >= CUTOFF:
        print(edge_names[i])
