'''
This script computes the main result of the paper
'''

import sys
import csv
import copy
import random
import pathlib
import statistics

sys.path.insert(1, '../../common')
from constants import *


# Read command line args
TEST_PROGRAMS = pathlib.Path(sys.argv[1])
TEST_CALLGRAPHS = pathlib.Path(sys.argv[2])
CONFIG_FILE = sys.argv[3]
METHOD_COUNT_RESULTS = pathlib.Path(sys.argv[4])

#Initialization
configs = read_config_file(CONFIG_FILE)
FULL_FILE = configs["FULL_FILE"]

def format_app_class(classname):
    return classname.replace(".","/")

def format_lib_class(classname):
    # delete first two characters. delete last 6 characters
    return classname[2:-6]

'''
READ APP AND LIB METHOD COUNTS
'''
app_method_counts = {}
lib_method_counts = {}

for prog_results in METHOD_COUNT_RESULTS.iterdir():
    with open(prog_results) as f:
        line1 = f.readline().strip()
        app_method_counts[prog_results.stem] = int(line1)
        line2 = f.readline().strip()
        lib_method_counts[prog_results.stem] = int(line2)
       

'''
READ LIB and APP CLASS LISTS
'''
lib_classes = {}
app_classes = {}

for test_program in TEST_PROGRAMS.iterdir():
    app_classes_file = test_program / "info" / "classes"
    app_set = set()
    with open(app_classes_file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            app_set.add(format_app_class(line))
    
    lib_classes_file = test_program / "info" / "libClasses"
    lib_set = set()
    with open(lib_classes_file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            lib_set.add(format_lib_class(line))
            
    app_classes[test_program.stem] = app_set
    lib_classes[test_program.stem] = lib_set

'''
READING TEST DATA AND COMPUTE COVERAGE SCORES
'''
coverage_scores = {
    "static_app":[],
    "static_overall":[],
    "dynamic_app":[],
    "dynamic_overall":[]
}


#print("Benchmark,Static_App_Coverage,Dynamic_App_Coverage,Static_Overall_Coverage,Dynamic_Overall_Coverage, Overall_Method_Count")
for test_prog in TEST_CALLGRAPHS.iterdir():
    test_file = test_prog / FULL_FILE
    # read all static and dynamic method names
    with open(test_file, 'r') as testfp:
        static_test_method_names = set()
        dynamic_test_method_names = set()
        csv_reader = csv.DictReader(testfp)
        for row in csv_reader:
            static_test_method_names.add(row[SRC_NODE_COL_NAME])
            static_test_method_names.add(row[DEST_NODE_COL_NAME])
            if (int(row[LABEL_COLUMN]) == POSITIVE_LABEL):
                dynamic_test_method_names.add(row[SRC_NODE_COL_NAME])
                dynamic_test_method_names.add(row[DEST_NODE_COL_NAME])   

    #compute the coverages
    static_app_methods = 0.0
    dynamic_app_methods = 0.0
    static_lib_methods = 0.0
    dynamic_lib_methods = 0.0
    
    for method in static_test_method_names:
        classname = method.split(".")[0]
        if classname in app_classes[test_prog.stem]:
            static_app_methods += 1
        elif classname in lib_classes[test_prog.stem]:
            static_lib_methods += 1
        elif method=="<boot>":
            continue
        else:
            print("ERROR: class not found:" + classname)
            print("Method was:" + method)
            print("Program was:" + test_prog.stem)
            print(app_classes[test_prog.stem])
            sys.exit(0)


    for method in dynamic_test_method_names:
        classname = method.split(".")[0]
        if classname in app_classes[test_prog.stem]:
            dynamic_app_methods += 1
        elif classname in lib_classes[test_prog.stem]:
            dynamic_lib_methods += 1
        elif method=="<boot>":
            continue
        else:
            print("ERROR: class not found:" + classname)
            print("Method was:" + method)
            print("Program was:" + test_prog.stem)
            print(app_classes[test_prog.stem])
            sys.exit(0)

    overall_method_count = (app_method_counts[test_prog.stem]
                  + lib_method_counts[test_prog.stem])
    stat_app_cov = static_app_methods / app_method_counts[test_prog.stem]
    dyn_app_cov = dynamic_app_methods / app_method_counts[test_prog.stem]
    stat_overall_cov = (static_app_methods + static_lib_methods) / overall_method_count
    dyn_overall_cov = (dynamic_app_methods + dynamic_lib_methods) / overall_method_count
    if stat_app_cov > stat_overall_cov and dyn_app_cov/stat_app_cov > 0.25 and dyn_app_cov > 0.1 and overall_method_count > 1000:       
        #print(f'{test_prog.stem},{stat_app_cov},{dyn_app_cov},{stat_overall_cov},{dyn_overall_cov},{overall_method_count}')
        print(test_prog.stem)
