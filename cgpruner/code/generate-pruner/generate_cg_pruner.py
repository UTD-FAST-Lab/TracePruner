'''
Main learning script.
'''

import numpy as np
import sys
import csv
import copy
import random
import pathlib
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn import tree

sys.path.insert(1, '../common')
from classifier_utils import *
from constants import *


'''
READING CONFIGS AND HEADERS
'''

# Read command line args
BENCHMARK_CALLGRAPHS = pathlib.Path(sys.argv[1])
TRAINING_PROGRAMS_LIST = pathlib.Path(sys.argv[2])
CONFIG_FILE = sys.argv[3]
LEARNED_MODEL_FILE = sys.argv[4]

#Initialization
configs = read_config_file(CONFIG_FILE)
SUBSAMPLE_FILE = configs["SUBSAMPLE_FILE"]
FULL_FILE = configs["FULL_FILE"]
SA_TRANS_HEADER = configs["SA_TRANS_HEADER"]
SA_DIRECT_HEADER = configs["SA_DIRECT_HEADER"]

header_names = compute_headers(BENCHMARK_CALLGRAPHS, SUBSAMPLE_FILE)
sanity_check_names(header_names, SA_TRANS_HEADER, SA_DIRECT_HEADER)

'''
READ TRAINING DATA
'''
x_train = []
y_train = []

#Get the list of training programs
with open(TRAINING_PROGRAMS_LIST) as f:
    training_progs = [line.rstrip() for line in f]

#Reading the training data
for prog in BENCHMARK_CALLGRAPHS.iterdir():
    if (not prog.stem in training_progs):
        continue
    training_file = prog / SUBSAMPLE_FILE
    with open(training_file, 'r') as trainfp:
        csv_reader = csv.DictReader(trainfp)
        for row in csv_reader:
            next_row = []
            for header in header_names:
                next_row.append(float(row[header]))
            x_train.append(next_row)
            y_train.append(int(row[LABEL_COLUMN]))

#Convert into array format
x_train = np.array(x_train)
y_train = np.array(y_train)

#Remove rows with static-analysis 0
x_train, y_train = remove_entries_with_static_0(
    x_train,y_train, SA_TRANS_INDEX, POSITIVE_LABEL)

'''CLASSIFIER TRAINING'''

clf = RandomForestClassifier(
    n_estimators = 1000,
    max_features = "sqrt",
    random_state = 0,
    max_depth=10,
    min_samples_split = 2,
    min_samples_leaf = 1,
    bootstrap = False,
    criterion = "entropy"
    )
clf = clf.fit(x_train, y_train)
joblib.dump(clf, LEARNED_MODEL_FILE)
print("Training Complete")
