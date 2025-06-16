# run doop with differnet configurations with 4 benchmark programs 

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import subprocess


doop_path = '/doop/doop-4.24.3/bin/doop'
benchmark_path = '/benchmark'


def read_configurations():
    config_path = 'configs/doop_default_1change_configs_v1.csv'
    # config_path = 'doop_pairwise_testcases.csv'
    return pd.read_csv(config_path).to_dict(orient='records')

def get_jar_files():
    jar_files = {}
    for program in os.listdir(benchmark_path):
        if os.path.isdir(os.path.join(benchmark_path, program)):
            jar_file = os.path.join(benchmark_path, program, 'final.jar')
            if os.path.exists(jar_file):
                jar_files[program] = jar_file
    return jar_files


def run_doop(program_name, jar_file, config, config_id):

    mainclass = 'Entrypoint'
    analysis_type = config['analysis']
    
    # Start building the command
    cmd_parts = [
        doop_path,
        "-i", jar_file,
        "-a", analysis_type,
        "--main", mainclass,
        "--id", f"{program_name}_{config_id}",
        "--dont-cache-facts",
        "--thorough-fact-gen",
        "-t", "180"
    ]

    for k, v in config.items():
        if k != 'analysis' and str(v).lower() == "true":
            cmd_parts.append(f"--{k}")

    try:
        subprocess.run(cmd_parts, timeout=60*60*3, check=True)
    except subprocess.TimeoutExpired:
        print(f"Timeout in config {config_id} for program {program_name}")
    except subprocess.CalledProcessError:
        print(f"DOOP failed for config {config_id} for program {program_name}")

def run_doop_parallel(num_threads, program, cid):

    configs = read_configurations()
    jar_files = get_jar_files()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for program_name, jar_file in jar_files.items():
            if program_name != program:
                continue
            for config_id, config in enumerate(configs):
                if config_id != cid:
                    continue
                print('running doop for program:', program_name, 'config_id:', config_id)
                print(config)
                futures.append(executor.submit(run_doop, program_name, jar_file, config, config_id))

        for future in futures:
           future.result()


def main():

    program= 'jasml'
    config_version = 'v1'
    config_id = 39

    run_doop_parallel(5, program, config_id)





if __name__ == "__main__":
    # Set the path to the Doop jar file
    main()