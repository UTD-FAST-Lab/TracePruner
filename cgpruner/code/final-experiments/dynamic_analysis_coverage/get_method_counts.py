'''
This is a script to run the null pointer analysis 
(with the full 0-CFA call-graph and with the reduced call-graph).
Please place this script in the analyses/wala folder to run.
'''
import sys
import pathlib
import os

TEST_PROJECTS_FOLDER = sys.argv[1]
RESULTS_FOLDER = sys.argv[2]

SKIP_COMPLETED = True
DRIVER_PROGRAM = "GetMethodCount"
JAVA_COMMAND = "java"
WALA_CORE_JAR = "com.ibm.wala.core-1.5.5.jar"
WALA_SHRIKE_JAR = "com.ibm.wala.shrike-1.5.5.jar"
WALA_UTIL_JAR = "com.ibm.wala.util-1.5.5.jar"
NPA_FOLDER = "../../wala-npa"
TIMEOUT = 7200
TIMEOUT_CMD = "timeout"

#add wala jars to classpath. Also add current folder
class_path_string = f'{NPA_FOLDER}/{WALA_CORE_JAR}:{NPA_FOLDER}/{WALA_SHRIKE_JAR}:{NPA_FOLDER}/{WALA_UTIL_JAR}:{NPA_FOLDER}:.:$CLASSPATH'
os.environ["CLASSPATH"] = class_path_string
os.system("echo $CLASSPATH")  

#create the output folder if it doesn't exist
results_folder_path = os.path.join(os.getcwd(),RESULTS_FOLDER)
if not os.path.exists(results_folder_path):
    os.mkdir(results_folder_path)

#Loop through projects
projects_folder_path = pathlib.Path(TEST_PROJECTS_FOLDER)
for project in projects_folder_path.iterdir():
    benchmark = project.name
    print(benchmark)
    if (SKIP_COMPLETED):
        if os.path.exists(os.path.join(results_folder_path,(benchmark+".txt"))):
            print("skipping completed benchmark.")
            continue
    benchmark_path = os.path.join(TEST_PROJECTS_FOLDER,benchmark)
    #skip non-directories
    if not os.path.isdir(benchmark_path):
        continue
    #Get jar file
    jarfile = ''
    for file in os.listdir(os.path.join(benchmark_path,"jarfile")):
        if file.endswith(".jar"):
            jarfile = file
    jarfile_path = os.path.join(benchmark_path,("jarfile/" + jarfile))
    app_classes_file = os.path.join(benchmark_path,("info/classes"))
    lib_classes_file = os.path.join(benchmark_path,("info/libClasses"))

    #execute wala on the jar
    wala_command = (JAVA_COMMAND
    + " " + DRIVER_PROGRAM
    + " -jarfile"
    + " " + jarfile_path
    + " -appClasses "
    + " " + app_classes_file
    + " -libClasses "
    + " " + lib_classes_file 
    + " > " +  results_folder_path
    + "/" + benchmark + ".txt"
    )
    os.system(wala_command)
