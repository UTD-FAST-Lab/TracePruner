'''
This is a script to run the null pointer analysis 
(with the full 0-CFA call-graph and with the reduced call-graph).
Please place this script in the analyses/wala folder to run.
'''
import sys
import pathlib
import os

PROJECTS_FOLDER = sys.argv[1]
RESULTS_FOLDER = sys.argv[2]
TEST_PROGRAMS_LIST = sys.argv[3]
if (len(sys.argv) > 4): # Using reduced call-graph
    REDUCED_CALLGRAPHS_FOLDER = sys.argv[4]
    USE_REDUCED_CALLGRAPHS = True
else:
    USE_REDUCED_CALLGRAPHS = False

SKIP_COMPLETED = True
DRIVER_PROGRAM = "nullpointeranalysis.WalaNullPointerAnalysis"
FILE_WITH_MAIN_CLASS = "info/mainclassname"
JAVA_COMMAND = "java"
WALA_CORE_JAR = "com.ibm.wala.core-1.5.5.jar"
WALA_SHRIKE_JAR = "com.ibm.wala.shrike-1.5.5.jar"
WALA_UTIL_JAR = "com.ibm.wala.util-1.5.5.jar"
NPA_FOLDER = "../../wala-npa"
TIMEOUT = 10800
TIMEOUT_CMD = "timeout"

#add wala jars to classpath. Also add current folder
class_path_string = f'{NPA_FOLDER}/{WALA_CORE_JAR}:{NPA_FOLDER}/{WALA_SHRIKE_JAR}:{NPA_FOLDER}/{WALA_UTIL_JAR}:{NPA_FOLDER}:$CLASSPATH'
os.environ["CLASSPATH"] = class_path_string

#create the output folder if it doesn't exist
results_folder_path = os.path.join(os.getcwd(),RESULTS_FOLDER)
if not os.path.exists(results_folder_path):
    os.mkdir(results_folder_path)

#Read the list of test programs
with open(TEST_PROGRAMS_LIST) as f:
    test_programs = [line.rstrip() for line in f]

#Loop through projects
projects_folder_path = pathlib.Path(PROJECTS_FOLDER)
for project in projects_folder_path.iterdir():
    benchmark = project.name
    if (benchmark not in test_programs):
        continue
    print(benchmark)
    if (SKIP_COMPLETED):
        if os.path.exists(os.path.join(results_folder_path,(benchmark+".txt"))):
            print("skipping completed benchmark.")
            continue
    benchmark_path = os.path.join(PROJECTS_FOLDER,benchmark)
    #skip non-directories
    if not os.path.isdir(benchmark_path):
        continue
    #Get jar file
    jarfile = ''
    for file in os.listdir(os.path.join(benchmark_path,"jarfile")):
        if file.endswith(".jar"):
            jarfile = file
    jarfile_path = os.path.join(benchmark_path,("jarfile/" + jarfile))
    #get main class name
    mainclassname_file = os.path.join(benchmark_path,FILE_WITH_MAIN_CLASS)
    with open(mainclassname_file) as fp:
        mainclass_name = fp.read().splitlines()[0]

    if USE_REDUCED_CALLGRAPHS:
        reduced_callgraph_option = (
            " -prunedCallgraph " 
            + REDUCED_CALLGRAPHS_FOLDER
            + "/" + str(benchmark) + ".txt"
        )
    else:
        reduced_callgraph_option = ""

    #execute wala on the jar
    wala_command = (TIMEOUT_CMD 
    + " " + str(TIMEOUT)
    + " " + JAVA_COMMAND
    + " " + DRIVER_PROGRAM
    + " -classpath"
    + " " + jarfile_path
    + " -mainclass"
    + " " + mainclass_name
    + reduced_callgraph_option
    + " > " +  results_folder_path
    + "/" + benchmark + ".txt"
    )
    os.system(wala_command)

