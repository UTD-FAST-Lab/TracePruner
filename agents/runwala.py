# mohammad rafieian


import os
import csv


TMP_DIR = "tmp"

WALA_DRIVER = "tools/wala/driver/WalaCallgraph.java"
WALA_CORE_JAR = "/home/mohammad/projects/cgPruner/CallGraphPruner/cgpruner/code/run-on-single-program/tools/wala/com.ibm.wala.core-1.5.9.jar"
WALA_SHRIKE_JAR = "/home/mohammad/projects/cgPruner/CallGraphPruner/cgpruner/code/run-on-single-program/tools/wala/com.ibm.wala.shrike-1.5.9.jar"
WALA_UTIL_JAR = "/home/mohammad/projects/cgPruner/CallGraphPruner/cgpruner/code/run-on-single-program/tools/wala/com.ibm.wala.util-1.5.9.jar"

AGENT = '/home/mohammad/projects/cgPruner/CallGraphPruner/agents/branch-j8/target/branch-j8-1.0-SNAPSHOT-jar-with-dependencies.jar'




def run_wala(program):
	 

	#Execute driver (wala) on jar file
    wala_driver_class = WALA_DRIVER[:-5] #remove .java
    RAW_WALA_OUTPUT = f'tmp/cgs.csv'
	
    mainclass = 'App'
    jar_file = 'home/mohammad/projects/cgPruner/CallGraphPruner/agents/sample_targets/variables/target/variables-1.0-SNAPSHOT.jar'

    command = f'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -javaagent:{AGENT} {wala_driver_class} -classpath {jar_file} -mainclass {mainclass} -output {RAW_WALA_OUTPUT} -resolveinterfaces true > test.txt'
	# -reflection false -analysis 0cfa

	
    os.system(command)




def main():
	run_wala()

	
	



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


