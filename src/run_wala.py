import os
import csv
import random
import subprocess as sp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Lock for thread-safe writing
write_lock = threading.Lock()

dataset_folder = "/home/mohammad/projects/callgraph-pruning/datasets/ml4cgp_study_data/xcorpus/xcorpus_jars_w_deps" 
csv_file = "/home/mohammad/projects/callgraph-pruning/datasets/configuration/1-way-eliminated.csv"
results_folder = "/home/mohammad/projects/callgraph-pruning/static-call-graphs/xcorpus"
wala_jar = '/home/mohammad/projects/callgraph-pruning/datasets/ml4cgp_study/ml4cg_SA/target/ml4cg_sa-1.0-SNAPSHOT-shaded.jar'
selected_program_file = "/home/mohammad/projects/callgraph-pruning/datasets/ml4cgp_study_data/xcorpus/xcorpus_sel_programs2.txt"


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
def run_wala(config, config_number):


    # Example: java -jar wala-driver.jar <jar_file> --reflectionSetting FULL --handleStaticInit TRUE ... -o <output_file>
    command = ["java", "-Xmx90g", "-cp", wala_jar, 'dev.c0pslab.analysis.CGGenRunner']+ ["-j", selected_program_file] +  ["-o", results_folder] + config.split()
    command += ['--confID', str(config_number)]
    # command = ["java", "-jar", wala_jar]+ ["--jars", jar_file] + ["-o", output_file]

    # Run the command
    proc = sp.Popen(command)

    # Wait for the process to complete
    proc.wait()

    print("Java program completed.")


    # return program_name, config_number, execution_time


# Function to execute WALA in parallel using multiple threads
# def run_wala_in_parallel(programs, configurations, num_threads):
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         futures = []
#         for config_number, config in enumerate(configurations, start=1):
#             for program in programs:
#                 program_name = program.split('/')[-1]
#                 jar_file = os.path.join(dataset_folder, program, f"{program_name}.jar")

#                 future = executor.submit(run_wala, program, jar_file, mainclass, config, config_number)
#                 futures.append(future)

#         # Process results as they complete
#         for future in as_completed(futures):
#             program_name, config_number, execution_time = future.result()
            
#             # Write results to file in a thread-safe manner
#             res_file = os.path.join(results_folder, 'information.txt')
#             with write_lock:
#                 with open(res_file, 'a') as file:  # Use 'a' for append mode
#                     file.write(f"Program: {program_name}, Config: {config_number}, Execution Time: {execution_time:.2f} seconds\n")

                    

def main():
    # Step 1: Read the configurations from the CSV file
    configurations = read_configurations(csv_file)

    # processed_programs = [folder for folder in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, folder))]

    # Step 2: Sample a subset of programs from the dataset  TODO: make it automatic but for now I will comment out this part
    # sampled_programs = sample_programs(dataset_folder, sample_size, processed_programs)


    # Step 3: Run WALA in parallel on the sampled programs
    # run_wala_in_parallel(selected_programs, configurations, max_threads)

    for config_number, config in enumerate(configurations, start=1):
        run_wala(config, config_number)

if __name__ == "__main__":
    main()
