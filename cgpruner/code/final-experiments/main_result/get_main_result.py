'''
This script computes the main result of the paper
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

sys.path.insert(1, '../../common')
from classifier_utils import *
from constants import *


'''
READING CONFIGS, HEADERS AND THE MODEL
'''

# Read command line args
BENCHMARK_CALLGRAPHS = pathlib.Path(sys.argv[1])
CONFIG_FILE = sys.argv[2]
LEARNED_MODEL_FILE = sys.argv[3]
TEST_PROGRAMS_LIST = sys.argv[4]

#Initialization
configs = read_config_file(CONFIG_FILE)
SUBSAMPLE_FILE = configs["SUBSAMPLE_FILE"]
FULL_FILE = configs["FULL_FILE"]
SA_TRANS_HEADER = configs["SA_TRANS_HEADER"]
SA_DIRECT_HEADER = configs["SA_DIRECT_HEADER"]
FANOUT_CONSTANT = float(configs["FANOUT_CONSTANT"])
DEST_NODE_IN_DEG_CONSTANT = float(configs["DEST_NODE_IN_DEG_CONSTANT"])
OUTPUT_CALLGRAPHS = configs["OUTPUT_CALLGRAPHS"]
CG_PRUNER_CALLGRAPHS = configs["CG_PRUNER_CALLGRAPHS"]
HEURISTIC_CALLGRAPHS = configs["HEURISTIC_CALLGRAPHS"]

#this is the threshold at which we will print the 
#new precision and recall. typically make it to be the
#equal precision-recall point.
PER_PROG_THRESHOLD = float(configs["PER_PROG_THRESHOLD"])


header_names = compute_headers(BENCHMARK_CALLGRAPHS, SUBSAMPLE_FILE)
sanity_check_names(header_names, SA_TRANS_HEADER, SA_DIRECT_HEADER)

clf = joblib.load(LEARNED_MODEL_FILE)  

'''
READING TEST DATA
'''
x_test = []
y_test = []
test_program_indices = []
test_idx = 0
file_ids = defaultdict()
test_src_method_names = []
test_dest_method_names = []
test_bytecode_offsets = []

#Read the test programs list
with open(TEST_PROGRAMS_LIST) as f:
    test_programs = [line.rstrip() for line in f]

for prog in BENCHMARK_CALLGRAPHS.iterdir():
    if prog.stem not in test_programs:
        continue
    test_idx += 1
    test_file = prog / SUBSAMPLE_FILE
    file_ids[test_idx] = prog
    with open(test_file, 'r') as testfp:
        csv_reader = csv.DictReader(testfp)
        for row in csv_reader:
            next_row = []
            for header in header_names:
                next_row.append(float(row[header]))
            x_test.append(next_row)
            y_test.append(int(row[LABEL_COLUMN]))
            test_program_indices.append(test_idx)
            test_src_method_names.append(row[SRC_NODE_COL_NAME])
            test_dest_method_names.append(row[DEST_NODE_COL_NAME])
            test_bytecode_offsets.append(row[BC_OFFSET_COL_NAME])

#Convert into array format
x_test = np.array(x_test)
y_test = np.array(y_test)
test_program_indices = np.array(test_program_indices)

'''
BASELINE
'''
print("PER_PROGRAM_BASELINE")
print("Benchmark,Precision,Recall")
precision,recall,fmeasure = compute_prog_lev_prec_rec(
        x_test[:,SA_TRANS_INDEX],y_test,test_program_indices,file_ids)
print("\nBASELINE")
print("Precision,Recall,F-measure")
print(str(np.round(precision,4)) + ","
     + str(np.round(recall,4)) + ","
     + str(np.round(fmeasure,4)))
print("")


'''
HEURISTICS
'''
print("HEURISTIC-RULE PRUNING")
print("Heuristic,Precision,Recall,F-measure")
y_half_heuristic = make_heuristic_prediction(x_test,header_names,FANOUT_CONSTANT, DEST_NODE_IN_DEG_CONSTANT, SA_TRANS_HEADER, SA_DIRECT_HEADER, False)
y_full_heuristic = make_heuristic_prediction(x_test,header_names,FANOUT_CONSTANT, DEST_NODE_IN_DEG_CONSTANT, SA_TRANS_HEADER, SA_DIRECT_HEADER, True)

precision_half,recall_half,fmeasure_half = compute_prog_lev_prec_rec(
        y_half_heuristic,y_test,test_program_indices,file_ids, False)
precision_full,recall_full,fmeasure_full = compute_prog_lev_prec_rec(
        y_full_heuristic,y_test,test_program_indices,file_ids, False)

print("half-heuristic,"
     + str(np.round(precision_half,4)) + ","
     + str(np.round(recall_half,4)) + ","
     + str(np.round(fmeasure_half,4)))
print("full-heuristic,"
     + str(np.round(precision_full,4)) + ","
     + str(np.round(recall_full,4)) + ","
     + str(np.round(fmeasure_full,4)))

'''
PREDICTION
'''

y_pred_proba = clf.predict_proba(x_test)
y_pred_proba = y_pred_proba[:,1] #just use probability of label 1
#Must convert all rows with all static analysis saying '0' to
#a '0' probability for the prediction - to remove the bias.
for i in range(len(x_test)):
    if x_test[i][SA_TRANS_INDEX]!=POSITIVE_LABEL:
        y_pred_proba[i]=0.0

#Calculate the precision recall curve values
print("\nNEW PER-PROG PRECISION-RECALL AT THRESHOLD:" + str(PER_PROG_THRESHOLD))
print("Benchmark,Precision,Recall")
precision,recall,thresholds = get_precision_recall_curve(
    y_pred_proba,y_test,test_program_indices,file_ids, PER_PROG_THRESHOLD)
final_auc = sklearn.metrics.auc(precision,recall)
print("\n Area under curve," + str(final_auc) + "\n")

#Just handling the corner case
if thresholds[0]==0.0:
    del thresholds[0]
    del precision[0]
    del recall[0]

#Print out the table
print("PRECISION-RECALL CURVE POINTS")
print("Probability Thresholds,Precision,Recall, F-measure")
for i in range(len(precision)-1):
    print(str(thresholds[i]) + ","
         + str(np.round(precision[i],4)) + ","
         + str(np.round(recall[i],4)) + ","
         + str(np.round(f_measure(precision[i],recall[i]),4)))


'''
COUNT MONOMORPHIC CALLS
'''
#Compute the new predictions at the given threshold
new_pred = []
for i in y_pred_proba:
    if i>PER_PROG_THRESHOLD: #This is the point we pick to talk about
        new_pred.append(1)
    else:
        new_pred.append(0)

#Count the monomorphic callsites for both the heuristic
# and the cg-pruner

iter_count = 0
for pred in [new_pred,y_full_heuristic]:
    iter_count += 1
    # Get the old and new fanout-counts
    old_fanout_counts = defaultdict(lambda: 0)
    new_fanout_counts = defaultdict(lambda: 0)
    da_fanout_counts = defaultdict(lambda: 0)
    callsites = set()

    for i in range(len(x_test)):
        (name, offset, test_prog_index) = (test_src_method_names[i], test_bytecode_offsets[i], test_program_indices[i])
        call_site = (name, offset, test_prog_index)
        callsites.add(call_site)
        if (x_test[i][SA_TRANS_INDEX]==1):
            old_fanout_counts[call_site] += 1
        if (pred[i]==1):
            new_fanout_counts[call_site] += 1
        if (y_test[i]==1):
            da_fanout_counts[call_site] += 1

    old_monomorphic_calls = []
    new_monomorphic_calls = []
    da_monomorphic_calls = []
    callsite_test_prog_indices = []

    for cs in callsites:
      (name, offset, test_prog_index) = cs
      callsite_test_prog_indices.append(test_prog_index)

      if old_fanout_counts[cs]==1:
        old_monomorphic_calls.append(1)
      else:
        old_monomorphic_calls.append(0)

      if new_fanout_counts[cs]==1:
        new_monomorphic_calls.append(1)
      else:
        new_monomorphic_calls.append(0)

      if da_fanout_counts[cs]==1:
        da_monomorphic_calls.append(1)
      else:
        da_monomorphic_calls.append(0)

    #Calculate and print the precision and recall in the 2 cases.
    new_precision, new_recall, new_fmeasure = compute_prog_lev_prec_rec(
            new_monomorphic_calls,da_monomorphic_calls,callsite_test_prog_indices,file_ids, False)

    old_precision, old_recall, old_fmeasure = compute_prog_lev_prec_rec(
            old_monomorphic_calls,da_monomorphic_calls,callsite_test_prog_indices,file_ids, False)

    print("\n MONOMORPHIC CALLS")
    if iter_count == 1:
        print("For CG-Pruner")
    else:
        print("For Heuristic")
    print("Old Precision-Recall:" + str(old_precision) + "," + str(old_recall))
    print("New Precision-Recall:" + str(new_precision) + "," + str(new_recall))

'''
OUTPUT CALLGRAPHS
'''
if OUTPUT_CALLGRAPHS == "True":
    for i in range(2):
        if i==0:
            output_folder_name = CG_PRUNER_CALLGRAPHS
            pred = new_pred
        else:
            output_folder_name = HEURISTIC_CALLGRAPHS
            pred = y_full_heuristic

        edge_set = defaultdict(set)
        for i in range(len(pred)):
            if pred[i] == 1:
                #include the edge in the edge-set
                edge_signature = f'{test_src_method_names[i]},{test_bytecode_offsets[i]},{test_dest_method_names[i]}'
                program_name = file_ids[test_program_indices[i]].stem
                edge_set[program_name].add(edge_signature)

        for (program, edge_signatures) in edge_set.items():
            with open(output_folder_name + "/" + program + ".txt", "w") as wp:
                for edge in edge_signatures:
                    wp.write(edge + "\n")
