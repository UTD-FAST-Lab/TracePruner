# mohammad rafieian


import os
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, subprocess
import argparse




WALA_DRIVER = "/home/mohammad/projects/CallGraphPruner/scripts/trace-generation/driver/wala-project/target/wala-project-1.0-SNAPSHOT-jar-with-dependencies.jar"
data_folder = '/home/mohammad/projects/CallGraphPruner_data'  
NJR1_DATASET_FOLDER = f'{data_folder}/njr-1_dataset/june2020_dataset'
PROGRAM_FILES = '/home/mohammad/projects/CallGraphPruner/scripts/trace-generation/programs.txt'

agent = '/home/mohammad/projects/CallGraphPruner/agents/edge-trace-agent/target/edge-trace-agent-1.0-SNAPSHOT-jar-with-dependencies.jar' 

agentLevel = (
	'cg', 
	'branch',
	'var'
)



def get_mainclass(program):
	mainclassname_path = os.path.join(NJR1_DATASET_FOLDER, program, 'info', "mainclassname")
	with open(mainclassname_path, 'r') as mainfile:
		mainclass = mainfile.readline().strip()
	return mainclass

def get_jar_file(program):
	jar_file = os.path.join(NJR1_DATASET_FOLDER, program, 'jarfile', f"{program}.jar")
	return jar_file


def run_wala(program, args):
	
	print(program)
		
	mainclass = get_mainclass(program)
	jar_file = get_jar_file(program)

	# Construct the command string

	if args.type == 'B':
		agentType = agentLevel[1]
		path = 'new_branches'
	elif args.type == 'C':
		agentType = agentLevel[0]
		path = 'new_cgs'

	output_dir = os.path.join('/home/mohammad/projects/CallGraphPruner/data', 'edge-traces', path, program)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	command = [
		'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java',
		f'-javaagent:{agent}=logLevel=method,agentLevel={agentType},output={output_dir}',
		'-jar', WALA_DRIVER,
		'-classpath', jar_file,
		'-mainclass', mainclass,
		'-reflection', 'false',
		'-analysis', '0cfa',
		'-resolveinterfaces', 'true'
	]

	command = ' '.join(command)
	os.system(command)


def run_wala_in_parallel(num_threads, programs, args):
	with ThreadPoolExecutor(max_workers=num_threads) as executor:
		futures = []
		for program in programs:		
			future = executor.submit(run_wala, program, args)


def main(args):
	
    programs = []
    with open(PROGRAM_FILES, 'r') as file:
        programs = [program.strip() for program in file.readlines()]
		
    run_wala_in_parallel(3, programs, args)

	

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Run WALA with different instrumentations")

	# Define command-line arguments
	parser.add_argument('--type', type=str, required=True, help="specify the type of instrumentation (B:Branch, C:Call graph)")

	args = parser.parse_args()  # Parse arguments


	main(args)
