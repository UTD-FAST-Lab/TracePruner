import os
import csv
import random
import subprocess as sp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Lock for thread-safe writing
write_lock = threading.Lock()


DS_NAME = 'xcorpus'    #options: njr1, xcorpus

ml4_data_path = '/20TB/mohammad/cg_dataset/ml4cgp_study_data'
ml4_study_path = '/home/mohammad/projects/cgPruner/WALADriver'
output_path = '/20TB/mohammad/cg_dataset/static_cg'
# njr1_jar_path = '/20TB/mohammad/cg_dataset/njr-1-dataset'

if DS_NAME == 'njr1':
    pass 
    # dataset_folder = f"{ml4_data_path}/xcorpus/xcorpus_jars_w_deps" 
    selected_program_file = f"{ml4_data_path}/njr1/cgs/njr1_programs_2.txt"
    results_folder = f"{output_path}/njr1"
elif DS_NAME == 'xcorpus':
    # dataset_folder = f"{ml4_data_path}/xcorpus/xcorpus_jars_w_deps" 
    selected_program_file = f"{ml4_data_path}/xcorpus/xcorpus_sel_programs2.txt"
    results_folder = f"{output_path}/xcorpus"
    
wala_jar = f'{ml4_study_path}/ml4cg_SA/target/ml4cg_sa-1.0-SNAPSHOT-shaded.jar'
csv_file = f"{ml4_study_path}/1-way-eliminated.csv"

is_default = True
WALA_DEFAULT_CONFIG = "--handleZeroLengthArray --handleStaticInit"

# Sample Size and Threads
max_threads = 10  # Number of threads to use for concurrent execution




# Read configurations from CSV file
def read_configurations(csv_file):
    configurations = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip the headers

        for row in reader:
            # --reflectionSetting, --handleStaticInit, --useConstantSpecificKeys, --cgalgo, --sensitivityو --handleZeroLengthArray,--useLexicalScopingForGlobals,--useStacksForLexcialScoping
            formatted_config = (
                f"--reflectionSetting={row[0]} "
                f"--cgalgo={row[3]} "
                f"--sensitivity={row[4]} "
            )
            if row[1] == 'TRUE':
                formatted_config += "--handleStaticInit "
            if row[2] == 'TRUE':
                formatted_config += "--useConstantSpecificKeys "
            if row[5] == 'TRUE':
                formatted_config += "--handleZeroLengthArray "
            if row[6] == 'TRUE':
                formatted_config += "--useLexicalScopingForGlobals "
            if row[7] == 'TRUE':
                formatted_config += "--useStacksForLexcialScoping"
            configurations.append(formatted_config)
  
    return configurations


# Function to sample programs from the dataset
def read_programs():
    programs = []

    with open(selected_program_file, 'r') as f1:
        lines = f1.readlines()

    for line in lines:
        programs.append(line.split(',')[0])

    return programs


# Function to run WALA with a configuration and a jar file
def run_wala(config='', config_number=0):


    # Example: java -jar wala-driver.jar <jar_file> --reflectionSetting FULL --handleStaticInit TRUE ... -o <output_file>
    command = ["/usr/lib/jvm/java-1.11.0-openjdk-amd64/bin/java", "-Xmx10g", "-cp", wala_jar, 'dev.c0pslab.analysis.CGGenRunner']+ ["-j", selected_program_file] +  ["-o", results_folder]
    command += ['--confID', str(config_number)]
    if not is_default:
        command += config.split()
    else:
        command += WALA_DEFAULT_CONFIG.split()

    # Run the command
    proc = sp.Popen(command)

    # Wait for the process to complete
    proc.wait()

    print("Java program completed.")


    # return program_name, config_number, execution_time


# Function to execute WALA in parallel using multiple threads
def run_wala_in_parallel(configurations, num_threads):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        if not is_default:
            for config_number, config in enumerate(configurations, start=1):
                future = executor.submit(run_wala, config, config_number)
                futures.append(future)
        else:
            future = executor.submit(run_wala)
            

def main():
    # Read the configurations from the CSV file
    configurations = read_configurations(csv_file)

    run_wala_in_parallel(configurations, max_threads)


if __name__ == "__main__":
    main()
