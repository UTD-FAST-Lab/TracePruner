# mohammad rafieian


import os
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, subprocess
import argparse




WALA_DRIVER = "/home/mohammad/projects/TracePruner/scripts/trace-generation/driver/wala-project/target/wala-project-1.0-SNAPSHOT-jar-with-dependencies.jar"
NJR1_DATASET_FOLDER = '/20TB/mohammad/njr-1-dataset/june2020_dataset'
PROGRAM_FILES = '/20TB/mohammad/data/programs.txt'

agent = '/home/mohammad/projects/TracePruner/agents/edge-trace-agent/target/edge-trace-agent-1.0-SNAPSHOT-jar-with-dependencies.jar' 

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

	output_dir = os.path.join('/20TB/mohammad/data', 'edge-traces-encode', path, program)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	command = [
		'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java',
		'-Xmx128g',
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

	# var_info_dir = os.path.join('/20TB/mohammad/data/variables', program)
	# if not os.path.exists(var_info_dir):
	# 	os.makedirs(var_info_dir)

	# var_path = os.path.join(var_info_dir, 'var.txt')
	# with open(var_path, 'w') as f:
	# 	process = subprocess.Popen(command, stdout=f, stderr=f, text=True)
	# 	process.communicate()


def run_wala_in_parallel(num_threads, programs, args):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_program = {
            executor.submit(run_wala, program, args): program
            for program in programs
        }

        for future in as_completed(future_to_program):
            program = future_to_program[future]
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error processing {program}: {e}")
            else:
                print(f"✅ Finished processing {program}")

def main(args):
	
    programs = []
    with open(PROGRAM_FILES, 'r') as file:
        programs = [program.strip() for program in file.readlines()]
		
    run_wala_in_parallel(10, programs, args)

	

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Run WALA with different instrumentations")

	# Define command-line arguments
	parser.add_argument('--type', type=str, required=True, help="specify the type of instrumentation (B:Branch, C:Call graph)")

	args = parser.parse_args()  # Parse arguments


	main(args)
