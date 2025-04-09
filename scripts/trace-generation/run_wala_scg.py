# mohammad rafieian


import os
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, subprocess
import argparse




WALA_DRIVER = "/home/mohammad/projects/CallGraphPruner/scripts/trace-generation/driver/wala-project_scg/target/wala-project_scg-1.0-SNAPSHOT-jar-with-dependencies.jar"
data_folder = '/home/mohammad/projects/CallGraphPruner_data'  
NJR1_DATASET_FOLDER = f'{data_folder}/njr-1_dataset/june2020_dataset'
PROGRAM_FILES = '/home/mohammad/projects/CallGraphPruner/scripts/trace-generation/programs.txt'



def get_mainclass(program):
	mainclassname_path = os.path.join(NJR1_DATASET_FOLDER, program, 'info', "mainclassname")
	with open(mainclassname_path, 'r') as mainfile:
		mainclass = mainfile.readline().strip()
	return mainclass

def get_jar_file(program):
	jar_file = os.path.join(NJR1_DATASET_FOLDER, program, 'jarfile', f"{program}.jar")
	return jar_file


def run_wala(program):
	
	print(program)
		
	mainclass = get_mainclass(program)
	jar_file = get_jar_file(program)
	output_file = f'/home/mohammad/projects/CallGraphPruner/data/static-cgs/{program}/wala0cfa.csv'

	# command = [
	# 	'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java',
	# 	'-jar', WALA_DRIVER,
	# 	'-classpath', jar_file,
	# 	'-mainclass', mainclass,
	# 	'-reflection', 'false',
	# 	'-analysis', '0cfa',
	# 	'-resolveinterfaces', 'true',
	# 	'-output', output_file
	# ]

	command = f'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -jar {WALA_DRIVER} -classpath {jar_file} -mainclass {mainclass} -output {output_file} -resolveinterfaces true -reflection false -analysis 0cfa '


	os.system(command)


def run_wala_in_parallel(num_threads, programs):
	with ThreadPoolExecutor(max_workers=num_threads) as executor:
		futures = []
		for program in programs:		
			future = executor.submit(run_wala, program)


def main():
	
    programs = []
    with open(PROGRAM_FILES, 'r') as file:
        programs = [program.strip() for program in file.readlines()]
		
    run_wala_in_parallel(15, programs)

	

if __name__ == '__main__':

	main()
