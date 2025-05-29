import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError
import subprocess

# Paths
WALA_DRIVER = "/home/mohammad/projects/TracePruner/scripts/static_cg_generation/wala/driver/wala-project_scg/target/wala-project_scg-1.0-SNAPSHOT-jar-with-dependencies.jar"
CONFIG_PATH = "/home/mohammad/projects/TracePruner/scripts/static_cg_generation/wala/configs/wala_1change_configs_v2.csv"

# Jar file map
jar_files = {
    'axion': '/20TB/mohammad/xcorpus-total-recall/jarfiles/axion/final.jar',
    'batik': '/20TB/mohammad/xcorpus-total-recall/jarfiles/batik/final.jar',
    'jasml': '/20TB/mohammad/xcorpus-total-recall/jarfiles/jasml/final.jar',
    'xerces': '/20TB/mohammad/xcorpus-total-recall/jarfiles/xerces/final.jar',
}

def read_configurations():
    return pd.read_csv(CONFIG_PATH).to_dict(orient='records')  # list of dicts

def run_wala(program_name, jar_file, config, config_id):
    mainclass = 'Entrypoint'

    print(f"Running for {program_name}: {config_id}")
    
    # Build unique output name from config
    config_suffix = "_".join(f"{k}_{config[k]}" for k in sorted(config))
    output_file = f'/20TB/mohammad/xcorpus-total-recall/static_cgs/wala/v2/{program_name}/wala_{config_id}.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Build command
    cmd_parts = [
        "/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java",
        "-jar", WALA_DRIVER,
        "-classpath", jar_file,
        "-mainclass", mainclass,
        "-output", output_file,
        "-resolveinterfaces", "true"
    ]
    for k, v in config.items():
        cmd_parts.extend([f"-{k}", str(v)])

    try:
        subprocess.run(cmd_parts, timeout=60*60*3, check=True)  # 3-hour timeout
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout in config {config_id} for program {program_name}")
    except subprocess.CalledProcessError:
        print(f"❌ WALA failed for config {config_id} for program {program_name}")

def run_wala_in_parallel(num_threads):
    configs = read_configurations()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for program_name, jar_file in jar_files.items():
            for i, config in enumerate(configs):
                futures.append(executor.submit(run_wala, program_name, jar_file, config, i))

        for future in futures:
            future.result()


def main():
    run_wala_in_parallel(6)

if __name__ == '__main__':
    main()