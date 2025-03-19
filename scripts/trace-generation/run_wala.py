# mohammad rafieian


import os
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, subprocess



WALA_DRIVER = "/home/mohammad/projects/CallGraphPruner/scripts/trace-generation/driver/wala-project/target/wala-project-1.0-SNAPSHOT-jar-with-dependencies.jar"
data_folder = '/home/mohammad/projects/CallGraphPruner_data'  
NJR1_DATASET_FOLDER = f'{data_folder}/njr-1_dataset/june2020_dataset'
PROGRAM_FILES = '/home/mohammad/projects/CallGraphPruner/scripts/trace-generation/programs.txt'

agent = '/home/mohammad/projects/CallGraphPruner/agents/integrated-edge-seperate-j8/target/integrated-edge-seperate-j8-1.0-SNAPSHOT-jar-with-dependencies.jar' 

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


def run_wala(program):
	
	print(program)
		
	mainclass = get_mainclass(program)
	jar_file = get_jar_file(program)

	# command = f'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -javaagent:{agent}=logLevel=method,agentLevel={agentLevel[1]} -jar {WALA_DRIVER} -classpath {jar_file} -mainclass {mainclass} -reflection false -analysis 0cfa -resolveinterfaces true'	
	# command += f' > /home/mohammad/projects/CallGraphPruner/data/traces/branches/{program}.txt'

	# Construct the command string
	command = [
		'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java',
		f'-javaagent:{agent}=logLevel=method,agentLevel={agentLevel[0]}',
		'-jar', WALA_DRIVER,
		'-classpath', jar_file,
		'-mainclass', mainclass,
		'-reflection', 'false',
		'-analysis', '0cfa',
		'-resolveinterfaces', 'true'
	]

	# Specify the output file for logs
	output_file = f'/home/mohammad/projects/CallGraphPruner/data/traces/cgs/{program}.txt'

	# Open the file to capture everything
	with open(output_file, 'w') as f:
		# Use Popen to capture stdout and stderr from the command
		process = subprocess.Popen(command, stdout=f, stderr=f, text=True)
		process.communicate()


	# os.system(command)


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
