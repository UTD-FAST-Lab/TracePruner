# mohammad rafieian


import os
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

TMP_DIR = "tmp"
# RAW_WALA_OUTPUT = "tmp/raw_wala_output.csv"
DECOMPILED_JSON = "tmp/decompile.json"
CALLSITES_FILE = "tmp/callsites.csv"
# METHODS_FILE = "tmp/methods.txt"
STD_LIB_FILE = "tmp/std-lib.txt"
FINAL_OUTPUT_FILE = "tmp/wala_callgraph.csv"
WALA_DRIVER = "tools/wala/driver/WalaCallgraph.java"
CALLSITES_SCRIPT = "tools/util/callsites.py"
CLOSE_GRAPH_SCRIPT = "tools/util/remove_std_lib_edges.py"
WALA_CORE_JAR = "tools/wala/com.ibm.wala.core-1.5.9.jar"
WALA_SHRIKE_JAR = "tools/wala/com.ibm.wala.shrike-1.5.9.jar"
WALA_UTIL_JAR = "tools/wala/com.ibm.wala.util-1.5.9.jar"


# data_folder = os.getenv('DATA_FOLDER')
data_folder = '/home/mohammad/projects/CallGraphPruner_data'    #change this to environment
OUTPUT_FOLDER = f'{data_folder}/output/static_cgs'
PROGRAM_FILES = '../programs.txt'
NJR1_DATASET_FOLDER = f'{data_folder}/njr-1_dataset/june2020_dataset'

# ml4_study_path = '/home/mohammad/projects/cgPruner/WALADriver'
csv_file = "../1-way-eliminated.csv"

is_default = False
WALA_DEFAULT_CONFIG = '-reflectionSetting NONE -cgalgo ZERO_CFA -sensitivity 1 -handleStaticInit true -useConstantSpecificKeys false -handleZeroLengthArray true -useLexicalScopingForGlobals false -useStacksForLexcialScoping false'

#Parse command line args
# p = argparse.ArgumentParser()
# p.add_argument("--output", help="File to output wala callgraph.")
# p.add_argument("--input", help="Input Jar file to run Wala on.")
# p.add_argument("--main", help="Name of the main class in the Jar")
# args = p.parse_args()



def get_mainclass(program):
	mainclassname_path = os.path.join(NJR1_DATASET_FOLDER, program, 'info', "mainclassname")
	with open(mainclassname_path, 'r') as mainfile:
		mainclass = mainfile.readline().strip()
	return mainclass

def get_jar_file(program):
	jar_file = os.path.join(NJR1_DATASET_FOLDER, program, 'jarfile', f"{program}.jar")
	return jar_file


# Read configurations from CSV file
def read_configurations(csv_file):
	configurations = []
	with open(csv_file, 'r') as file:
		reader = csv.reader(file)
		headers = next(reader)  # Skip the headers

		for row in reader:
            # --reflectionSetting, --handleStaticInit, --useConstantSpecificKeys, --cgalgo, --sensitivityو --handleZeroLengthArray,--useLexicalScopingForGlobals,--useStacksForLexcialScoping
			formatted_config = (
				f"-reflectionSetting {row[0]} "
				f"-cgalgo {row[3]} "
				f"-sensitivity {row[4]} "
			)
			if row[1] == 'TRUE':
				formatted_config += "-handleStaticInit true "
			else:
				formatted_config += "-handleStaticInit false "
			if row[2] == 'TRUE':
				formatted_config += "-useConstantSpecificKeys true "
			else:
				formatted_config += "-useConstantSpecificKeys false "
			if row[5] == 'TRUE':
				formatted_config += "-handleZeroLengthArray true "
			else:
				formatted_config += "-handleZeroLengthArray false "
			if row[6] == 'TRUE':
				formatted_config += "-useLexicalScopingForGlobals true "
			else:
				formatted_config += "-useLexicalScopingForGlobals false "
			if row[7] == 'TRUE':
				formatted_config += "-useStacksForLexcialScoping true"
			else:
				formatted_config += "-useStacksForLexcialScoping false"
			configurations.append(formatted_config)
  
	return configurations

def run_wala(program, config='', config_num=0):
	 
	mainclass = get_mainclass(program)
	jar_file = get_jar_file(program)

	if not os.path.exists(os.path.join(OUTPUT_FOLDER, program)):
		os.makedirs(os.path.join(OUTPUT_FOLDER, program))

	
	output_file = os.path.join(OUTPUT_FOLDER, program, f'{program}_config_{config_num}.csv')


	#Execute driver (wala) on jar file
	wala_driver_class = WALA_DRIVER[:-5] #remove .java
	RAW_WALA_OUTPUT = f'tmp/raw_wala_output_{config_num}_{program}.csv'

	command = f'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java {wala_driver_class} -classpath {jar_file} -mainclass {mainclass} -output {RAW_WALA_OUTPUT} -resolveinterfaces true '
	# -reflection false -analysis 0cfa

	if not is_default:
		command += config
	else:
		command += WALA_DEFAULT_CONFIG
	
	os.system(command)

	# Run javaq to get methods
	METHODS_FILE = f'tmp/methods_{config_num}_{program}.txt'
	os.system(f'/home/mohammad/projects/CallGraphPruner/cgpruner/javaq/.stack-work/install/x86_64-linux-tinfo6/957f5142b7e8110d4a9248773920af363e8c0e689145f498a5a9416c55915651/8.10.4/bin/javaq --cp {jar_file} list-methods > {METHODS_FILE}')

	#Finally compute the transitive closure
	os.system(f'python3 {CLOSE_GRAPH_SCRIPT} {RAW_WALA_OUTPUT} {METHODS_FILE} {output_file}')


def run_wala_in_parallel(configurations, num_threads, programs):
	with ThreadPoolExecutor(max_workers=num_threads) as executor:
		futures = []
		for program in programs:
			if not is_default:
				for config_number, config in enumerate(configurations, start=1):
					future = executor.submit(run_wala,program, config, config_number)
					futures.append(future)
			else:
				future = executor.submit(run_wala, program)



def main():

	configurations = read_configurations(csv_file)
	programs = []
	with open(PROGRAM_FILES, 'r') as file:
		programs = [program.strip() for program in file.readlines()]

	run_wala_in_parallel(configurations, 10, programs)
	
	



if __name__ == '__main__':
    #Create temporary directory if it doesn't exist
	if not os.path.exists(TMP_DIR):
		os.makedirs(TMP_DIR)

	#Add jars to classpath
	class_path_string = f'{WALA_CORE_JAR}:{WALA_SHRIKE_JAR}:{WALA_UTIL_JAR}:.'
	os.environ["CLASSPATH"] = class_path_string

	#compile java driver
	os.system(f'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/javac {WALA_DRIVER}')
	
	main()


'''
RAW COMMANDS
python run_wala --output arg.output --input arg.jar --main arg.main
export CLASSPATH=tools/wala/com.ibm.wala.shrike-1.5.5.jar:tools/wala/com.ibm.wala.core-1.5.5.jar:tools/wala/com.ibm.wala.util-1.5.5.jar:$CLASSPATH
javac tools/wala/driver/WalaCallgraph.java
java tools.wala.driver.WalaCallgraph -classpath sample-program/jarfile/jar1.jar -mainclass praxis.PraxisController -output out.csv -reflection false -analysis 0cfa -resolveinterfaces true
javaq --cp sample-program/jarfile/jar1.jar list-methods > methods.txt
python3 tools/util/remove_std_lib_edges.py out.csv methods.txt wala_callgraph.csv
'''
