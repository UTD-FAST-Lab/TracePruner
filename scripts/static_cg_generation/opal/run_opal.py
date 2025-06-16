import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import subprocess

# Paths
OPAL_DRIVER = "/OPALInterface/target/OPAL-1.0-SNAPSHOT-jar-with-dependencies.jar"
CONFIG_PATH = "opal_pairwise_testcases.csv"

benchmark_path = '/benchmark'


def read_configurations():
    return pd.read_csv(CONFIG_PATH).to_dict(orient='records')  # list of dicts


def get_jar_files():
    jar_files = {}
    for program in os.listdir(benchmark_path):
        if os.path.isdir(os.path.join(benchmark_path, program)):
            jar_file = os.path.join(benchmark_path, program, 'final.jar')
            if os.path.exists(jar_file):
                jar_files[program] = jar_file
    return jar_files



def run_opal(program_name, jar_file, config, config_id):
    mainclass = 'Entrypoint'
    
    # Build unique output name from config
    output_file = f'/data/v1/{program_name}/opal_{config_id}.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    algorithm = config['algorithm']

    # Build command parts
    cmd_parts = [
        "java",
        "-Xmx20g",
        "-jar", OPAL_DRIVER,
        jar_file,
        output_file,
        algorithm
    ]

    try:
        subprocess.run(cmd_parts, timeout=60 * 60 * 3, check=True)
    except subprocess.TimeoutExpired:
        print(f"Timeout in OPAL config {config_id} for program {program_name}")
    except subprocess.CalledProcessError:
        print(f"OPAL failed for config {config_id} for program {program_name}")


def run_opal_in_parallel(num_threads):
    configs = read_configurations()
    jar_files = get_jar_files()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for program_name, jar_file in jar_files.items():
                for config_id, config in enumerate(configs):
                    futures.append(executor.submit(run_opal, program_name, jar_file, config, config_id))

        for future in futures:
            future.result()


def main():
    run_opal_in_parallel(2)

if __name__ == '__main__':
    main()

