
import os
import argparse

TMP_DIR = "tmp"
RAW_WALA_OUTPUT = "tmp/raw_wala_output.csv"
DECOMPILED_JSON = "tmp/decompile.json"
CALLSITES_FILE = "tmp/callsites.csv"
METHODS_FILE = "tmp/methods.txt"
STD_LIB_FILE = "tmp/std-lib.txt"
FINAL_OUTPUT_FILE = "tmp/wala_callgraph.csv"
WALA_DRIVER = "tools/wala/driver/WalaCallgraph.java"
CALLSITES_SCRIPT = "tools/util/callsites.py"
CLOSE_GRAPH_SCRIPT = "tools/util/remove_std_lib_edges.py"
WALA_CORE_JAR = "tools/wala/com.ibm.wala.core-1.5.5.jar"
WALA_SHRIKE_JAR = "tools/wala/com.ibm.wala.shrike-1.5.5.jar"
WALA_UTIL_JAR = "tools/wala/com.ibm.wala.util-1.5.5.jar"

#Parse command line args
p = argparse.ArgumentParser()
p.add_argument("--output", help="File to output wala callgraph.")
p.add_argument("--input", help="Input Jar file to run Wala on.")
p.add_argument("--main", help="Name of the main class in the Jar")
args = p.parse_args()

#Create temporary directory if it doesn't exist
if not os.path.exists(TMP_DIR):
	os.makedirs(TMP_DIR)

#Add jars to classpath
class_path_string = f'{WALA_CORE_JAR}:{WALA_SHRIKE_JAR}:{WALA_UTIL_JAR}:.'
os.environ["CLASSPATH"] = class_path_string

#compile java driver
os.system(f'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/javac {WALA_DRIVER}')

#Execute driver (wala) on jar file
wala_driver_class = WALA_DRIVER[:-5] #remove .java
os.system(f'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java {wala_driver_class} -classpath {args.input} -mainclass {args.main} -output {RAW_WALA_OUTPUT} -reflection false -analysis 0cfa -resolveinterfaces true')

#Run javaq to get methods
# os.system(f'javaq --cp {args.input} list-methods > {METHODS_FILE}')

# #Finally compute the transitive closure
# os.system(f'python3 {CLOSE_GRAPH_SCRIPT} {RAW_WALA_OUTPUT} {METHODS_FILE} {args.output}')

'''
RAW COMMANDS
python run_wala --output arg.output --input arg.jar --main arg.main
export CLASSPATH=tools/wala/com.ibm.wala.shrike-1.5.5.jar:tools/wala/com.ibm.wala.core-1.5.5.jar:tools/wala/com.ibm.wala.util-1.5.5.jar:$CLASSPATH
javac tools/wala/driver/WalaCallgraph.java
java tools.wala.driver.WalaCallgraph -classpath sample-program/jarfile/jar1.jar -mainclass praxis.PraxisController -output out.csv -reflection false -analysis 0cfa -resolveinterfaces true
javaq --cp sample-program/jarfile/jar1.jar list-methods > methods.txt
python3 tools/util/remove_std_lib_edges.py out.csv methods.txt wala_callgraph.csv
'''
