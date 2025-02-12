'''
SUPPLEMENTARY FUNCTIONS NEEDED FOR MAIN 
learn.py CLASSIFICATIOIN SCRIPT
'''

import numpy as np
import statistics
import sklearn.metrics
import random
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from collections import defaultdict

from constants import *


'''
Represents a program's edges with its expected labels, .
Mainly used for computing the per-program precision/recall
'''
class TestProgram:
    def __init__(self):
        self.prog_predicted_labels = []
        self.prog_expected_labels = []
        self.prog_precicted_probabilities = []
        self.prog_learned_labels = []
        self.callgraph_size = 0
        self.precision_values = None
        self.recall_values = None
        self.thresholds = None

def compute_prog_lev_prec_rec(prediction,truth,test_program_indices,file_ids, print_per_prog_values=True):
    precision_values = []
    recall_values = []
    f_measure_values = []

    #First group all the edges from one program together
    programs = defaultdict(lambda: TestProgram())
    for i in range(len(test_program_indices)):
        programs[test_program_indices[i]].prog_expected_labels.append(truth[i])
        programs[test_program_indices[i]].prog_predicted_labels.append(prediction[i])

    #Next, compute the precision, recall, threshold values for every program.
    for prog_index,prog in programs.items():
        if (sum(prog.prog_predicted_labels)==0):
            print("Skipping:" + file_ids[prog_index].stem)
            continue
        precision,recall,f_measure = compute_precision_recall(
            prog.prog_predicted_labels,prog.prog_expected_labels)
        precision_values.append(precision)
        recall_values.append(recall)
        f_measure_values.append(f_measure)

        #Print the values if necessary
        if print_per_prog_values:
            print(file_ids[prog_index].name
                + "," + str(round(precision,4))
                + "," + str(round(recall,4))
            )    #+ "," + str(len(prog.prog_predicted_labels))
                #+ "," + str(sum(prog.prog_predicted_labels))
            #)

    return (
        statistics.mean(precision_values),
        statistics.mean(recall_values), 
        statistics.mean(f_measure_values)
        )

def compute_precision_recall(prediction,truth):
    tp = 0.0
    fp = 0.0
    fn = 0.0

    for i in range(len(prediction)):
        if (prediction[i]==1 and truth[i]==1):
            tp += 1
        
        if (prediction[i]==1 and truth[i]==0):
            fp += 1
        
        if ( prediction[i]==0 and truth[i]==1):
            fn += 1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision,recall,f_measure(precision,recall)

def f_measure(precision,recall):
    if (precision==0 or recall==0):
        return 0
    return 2.0 / ((1.0/precision) + (1.0/recall))


def get_precision_recall_curve(y_pred_proba,y_test,test_program_indices,file_ids,per_prog_threshold=0.4, montonic_precision_values=True):
    '''Returns a set of precision-recall tradeoff points'''

    if PROGRAM_LEVEL_EVAL: #Correctly compute the precision/recall per program
        precision, recall, thresholds = per_program_precision_recall_curve(y_test, y_pred_proba,test_program_indices,file_ids,per_prog_threshold)
    else:
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
            y_test, y_pred_proba,pos_label=1)
        #Adding corner case of prob=1.0 so that the 3 arrays have the same size
        thresholds = np.append(thresholds,1.0)

    if (montonic_precision_values): #Use only the increasing subsequence of precision values
        previous_highest = 0
        final_precisions = []
        final_recalls = []
        final_thresholds = []
        for i in range(len(precision)):
            if precision[i]>=previous_highest:
                previous_highest = precision[i]
                final_precisions.append(precision[i])
                final_recalls.append(recall[i])
                final_thresholds.append(thresholds[i])
        #Add a point on the Y-axis, to the left of the first point. Better area calculation
        final_precisions.insert(0,0)
        final_recalls.insert(0,final_recalls[0])
        final_thresholds.insert(0,0.00000000000001)
        return final_precisions,final_recalls,final_thresholds
    else: #Just return whatever we have
        return precision, recall, thresholds


def per_program_precision_recall_curve(expected_labels, predicted_probs, test_program_indices,file_ids, per_prog_threshold):
    '''Compute the precision and recall per program instead of over all the edges
    '''
    avg_precision_values = []
    avg_recall_values = []
     
    #First group all the edges from one program together
    programs = defaultdict(lambda: TestProgram())
    for i in range(len(test_program_indices)):
        programs[test_program_indices[i]].prog_expected_labels.append(expected_labels[i])
        programs[test_program_indices[i]].prog_precicted_probabilities.append(predicted_probs[i])

    #Next, compute the precision, recall, threshold values for every program.
    thresholds_union_set = set()
    for prog_idx,prog in programs.items():
        prog.precision_values,prog.recall_values,prog.thresholds = (
            sklearn.metrics.precision_recall_curve(prog.prog_expected_labels, prog.prog_precicted_probabilities,pos_label=1))
        thresholds_union_set.update(prog.thresholds)
        #Adding the corner case of prob=1.0 so that the sizes 
        #of the 3 arrays are same
        prog.thresholds = np.append(prog.thresholds,1.0)

    #Print the per-program values if necessary
    for prog_idx,prog in programs.items():
        for i in range(len(prog.thresholds)):
            if prog.thresholds[i] > per_prog_threshold:
                print(str(file_ids[prog_idx])
                    + "," + str(round(prog.precision_values[i],4))
                    + "," + str(round(prog.recall_values[i],4))
                )
                break

    thresholds = list(thresholds_union_set)
    thresholds.sort()
    #Finally compute the avg precision and avg recall values
    for thresh in thresholds:
        per_program_precisions = []
        per_program_recalls = []
        #Remember that each program's thresholds will be a subset of the 'thresholds' list here
        #So we need to account for that.
        for p_,prog in programs.items():
            if thresh>prog.thresholds[-1]: #corner case. 
                #This program has no edges at such a high probability
                per_program_recalls.append(0)
                per_program_precisions.append(1.0)
            else:
                prog_pre = None
                prog_rec = None
                #Find the precision/recall value at this 'thresh' value for this program
                for i in range(len(prog.thresholds)):
                    #Check if we found the matching value
                    if prog.thresholds[i]>=thresh:
                        prog_pre = prog.precision_values[i]
                        prog_rec = prog.recall_values[i]
                        break
                per_program_precisions.append(prog_pre)
                per_program_recalls.append(prog_rec)

        #Take an average over all the programs.
        avg_precision_values.append(statistics.mean(per_program_precisions))
        avg_recall_values.append(statistics.mean(per_program_recalls))

    return avg_precision_values,avg_recall_values,thresholds


def kfold_cross_val(x, y, program_indices,classifier, folds ,neural_network_classifier,b_size):
    '''Calculates average area under precision-recall curve for 
    k-fold cross validation, but splits are done at the program granularity.
    '''
    area_under_curve_scores = [] #final scores for each fold
    unique_program_indices = np.unique(program_indices)
    np.random.shuffle(unique_program_indices) #So that the next step gives random folds
    #Is a 2d array. Each row corresponds to a fold, and the elements in the 
    #row signify the programs in a fold
    fold_indices = np.array_split(unique_program_indices, folds)

    #In each iteration, a different fold is set as the test set
    for cross_val_test_indices_set in fold_indices:
        cross_val_x_train = []
        cross_val_x_test = []
        cross_val_y_train = []
        cross_val_y_test = []
        cv_test_indices = []
        
        #In this iteration, cross_val_test_indices_set is the test set
        #Everything else is the train set
        for i in range(len(program_indices)):
            if program_indices[i] in cross_val_test_indices_set:
                cross_val_x_test.append(x[i])
                cross_val_y_test.append(y[i])
                cv_test_indices.append(program_indices[i])
            else:
                cross_val_x_train.append(x[i])
                cross_val_y_train.append(y[i])

        #convert to numpy format
        cross_val_x_train = np.array(cross_val_x_train)
        cross_val_x_test = np.array(cross_val_x_test)
        cross_val_y_train = np.array(cross_val_y_train)
        cross_val_y_test = np.array(cross_val_y_test)
        cv_test_indices = np.array(cv_test_indices)

        #Do the training and testing for this split
        if neural_network_classifier:
            classifier.fit(cross_val_x_train, cross_val_y_train, epochs=NN_EPOCHS, batch_size=b_size,verbose=0)
            y_pred_proba = classifier.predict(cross_val_x_test)
        else:
            clf = classifier.fit(cross_val_x_train,cross_val_y_train)
            y_pred_proba = clf.predict_proba(cross_val_x_test)

        #Now compute the precision-recall area-under-curve for this split.
        precision_values,recall_values,t = get_precision_recall_curve(
            cross_val_x_test,y_pred_proba,cross_val_y_test,cv_test_indices)
        
        #Add a point with 0 precision to the right of the first point
        precision_values.insert(0,0)
        recall_values.insert(0,recall_values[0])

        if len(precision_values)<2:
            print("Error: No point returned by get_precision_recall_curve function")
        area_under_curve = sklearn.metrics.auc(precision_values,recall_values)
        area_under_curve_scores.append(area_under_curve)
        
    return statistics.mean(area_under_curve_scores)


def train_validation_score(x, y, program_indices,classifier,neural_network_classifier,b_size):
    '''Calculates average area under precision-recall curve for 
    k-fold cross validation, but splits are done at the program granularity.
    '''
    x_train,x_val,y_train,y_val,train_program_indices,val_program_indices=(
        get_train_test_split(x, y, program_indices, train_size=0.7))

    #Do the training and testing for this split
    if neural_network_classifier:
        classifier.fit(x_train, y_train, epochs=NN_EPOCHS, batch_size=b_size,verbose=0)
        y_pred_proba = classifier.predict(x_val)
    else:
        clf = classifier.fit(x_train,y_train)
        y_pred_proba = clf.predict_proba(x_val)

    #Now compute the precision-recall area-under-curve for this split.
    precision_values,recall_values,t = get_precision_recall_curve(
        x_val,y_pred_proba,y_val,val_program_indices)
    
    #Add a point with 0 precision to the right of the first point
    precision_values.insert(0,0)
    recall_values.insert(0,recall_values[0])

    if len(precision_values)<2:
        print("Error: No point returned by get_precision_recall_curve function")
    area_under_curve = sklearn.metrics.auc(precision_values,recall_values)
    return area_under_curve



def hyperparameter_optimized_random_forest(x_train,y_train,train_program_indices):
    '''Tries different hyperparameters for the random forest
    and returns the best classifier out of them based on the 
    one with the highest kfold_cross_val score'''
    n_estimators_options = [5,25,100]
    max_features_options = ['auto', 'sqrt']
    max_depth_options = [5,10,15,40,None]
    min_samples_split_options = [2, 5, 10]
    min_samples_leaf_options = [1, 2, 4]
    bootstrap_options = [True, False]
    criterion_options = ["gini","entropy"]

    best_classifier_so_far = RandomForestClassifier()
    best_cv_score_so_far = -1
    corresponding_hyperparameters = {}
    
    #Try num_configs many random combinations of options
    for i in range(NUM_HYPERPARAMETER_CONFIGS_TO_TRY):
        #Choose a random set of parameters
        n_estimators = random.choice(n_estimators_options)
        max_features = random.choice(max_features_options)
        max_depth = random.choice(max_depth_options)
        min_samples_split = random.choice(min_samples_split_options)
        min_samples_leaf = random.choice(min_samples_leaf_options)
        bootstrap = random.choice(bootstrap_options)
        criterion = random.choice(criterion_options)
        #Create a random forest with these parameters
        clf = RandomForestClassifier(random_state=0,
            n_estimators = n_estimators,
            max_features = max_features,
            max_depth = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            bootstrap = bootstrap,
            criterion = criterion
        )
        #Calculate cross validation score
        cv_score = kfold_cross_val(x_train, y_train, train_program_indices,
            clf, 3,False,0)
        #Record classifier, cv_score and hyperparameters if this has been the best
        if cv_score>best_cv_score_so_far:
            best_classifier_so_far = clf
            best_cv_score_so_far = cv_score
            corresponding_hyperparameters = { 
                "n_estimators" : n_estimators,
                "max_features" : max_features,
                "max_depth" : max_depth,
                "min_samples_split" : min_samples_split,
                "min_samples_leaf" : min_samples_leaf,
                "bootstrap" : True,
                "criterion" : criterion
            }
    return best_classifier_so_far, corresponding_hyperparameters



#Helper function for permutation importance
def permute_and_get_new_predictions(entry,x_test,header_names,clf):
    x_test_copy = x_test.copy() #preserve old x_test as is
    #Split the array into sub-arrays of columns 
    num_cols = np.size(x_test_copy,axis=1)
    x_split = np.split(x_test_copy,num_cols,axis=1)

    #Shuffle the columns having the 'entry' variable as a substring
    for i in range(len(header_names)):
        if (entry in header_names[i]): #this column has to be shuffled
            np.random.shuffle(x_split[i])
    
    #Join the columns back into the final 2d matrix
    new_x_test = np.concatenate(x_split,axis=1)

    #Now recompute the area under the curve, with this new test dataset
    if NUERAL_NETWORK:
        new_y_pred_proba = clf.predict(new_x_test)
    else:
        new_y_pred_proba = clf.predict_proba(new_x_test)
    return new_y_pred_proba

'''
Calculates the permutation importance for every entry in the dictionary.
We are using this instead of the ELI5 library implementation of it because
we want to remove all the features in an analysis/extra-feature at once.
'''
def get_permutation_importance_scores(dictionary, x_test, clf, old_auc, unscaled_x_test, y_test, test_program_indices, header_names):
    #"dictionary" could either be a dictionary of analyses of extra_features
    for entry in dictionary:
        new_y_pred_proba = permute_and_get_new_predictions(entry,x_test,header_names,clf)
        new_prec,new_recall,new_thresh = get_precision_recall_curve(
            unscaled_x_test,new_y_pred_proba,y_test,test_program_indices)
        new_auc = sklearn.metrics.auc(new_prec,new_recall)

        #Record the importance for that analysis/extra-feature
        dictionary[entry] = old_auc - new_auc

#removes all entries for which all static analyses say 0
def remove_entries_with_static_0(x,y,sa_trans_index,positive_label):
    row_indices_with_all0 = []
    for i in range(len(x)):
        if x[i][sa_trans_index]!=positive_label:
            row_indices_with_all0.append(i)
    x = np.delete(x, row_indices_with_all0, axis=0)
    y = np.delete(y, row_indices_with_all0, axis=0)
    return x,y


def make_heuristic_prediction(x_test,header_names, fanout_constant, dest_node_in_deg_constant,sa_trans_header, sa_direct_header, full_heuristic):
    '''makes a predicition for the test set based on a
    simple heuristic derived from examinging the decision tree
    '''
    prediction = []
    fanout_header = sa_trans_header + "#fanout"
    dest_node_in_deg_header = sa_trans_header + "#dest_node_in_deg"
    depth_header = sa_direct_header + "#depth_from_main"
    fanout_index = header_names.index(fanout_header)
    dest_node_in_deg_index = header_names.index(dest_node_in_deg_header)
    depth_index = header_names.index(depth_header)

    for e in x_test:
        #decide which heuristic to use
        if full_heuristic:
            heuristic = ((e[fanout_index] > fanout_constant and e[dest_node_in_deg_index] > dest_node_in_deg_constant)
            or (e[depth_index]>100000))
        else:
            heuristic = e[fanout_index] > fanout_constant and e[dest_node_in_deg_index] > dest_node_in_deg_constant

        #Mark false-positives based on this heuristic
        if heuristic:
            prediction.append(0) #it is a false positive
        else:
            prediction.append(e[SA_TRANS_INDEX]) #else return the static analysis output
    return np.array(prediction)

def compute_jaccard_similarity(arr1,arr2):
    '''Computes the jaccard similarity between
    2 sets represented as binary arrays'''
    intersect = 0.0
    union = 0.0
    for i in range(len(arr1)):
        if arr1[i]==1 and arr2[i]==1:
            intersect += 1
        if arr1[i]==1 or arr2[i]==1:
            union += 1    
    return intersect/union

def get_method_name(full_name):
    if full_name=="<boot>":
        return "<boot>"
    else:
        return full_name.split(".")[1]

def get_class_name(full_name):
    if full_name=="<boot>":
        return "<boot>"
    else:
        return full_name.split(".")[0]


#Returns the statistics on the cascading and overapproximating edges.
def get_cascading_stats(pred,y_test,src_method_names,dest_method_names):
    true_pos = 0
    overappr = 0
    cascading_over_appr = 0
    reachable_methods = set()

    #compute the set of reachable methods
    for i in range(len(y_test)):
        if y_test[i]==1:
            reachable_methods.add(src_method_names[i])
            reachable_methods.add(dest_method_names[i])

    #Now loop through the edges to classify them into the 3 buckets
    for i in range(len(pred)):
        if pred[i]==1 and y_test[i]==1:
            true_pos += 1
        elif pred[i]==1 and y_test[i]==0: #false-positive
            if src_method_names[i] in reachable_methods:
                overappr += 1
            else:
                cascading_over_appr += 1

    return true_pos, overappr, cascading_over_appr
