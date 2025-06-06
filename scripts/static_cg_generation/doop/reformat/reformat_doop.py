
import csv
import re
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess

extractor_jar_path = '/home/mohammad/projects/TracePruner/scripts/static_cg_generation/doop/reformat/DoopRunnerWithPC/target/DoopRunnerWithPC-1.0-SNAPSHOT-jar-with-dependencies.jar'
jdkPath = '/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/rt.jar'


def process_program(input_programs_dir, output_dir, program_dir):
    program_input_path = os.path.join(input_programs_dir, program_dir)
    input_file = os.path.join(program_input_path, 'CallGraphEdge.csv')

    program_name = program_dir.split('_')[0]
    inputJar = f'/20TB/mohammad/xcorpus-total-recall/jarfiles/{program_name}/final.jar'


    output_file_dir = os.path.join(output_dir, program_dir)
    os.makedirs(output_file_dir, exist_ok=True)
    output_file = os.path.join(output_file_dir, 'CallGraphEdge.csv')

    print(f"[+] Processing {input_file}...")

    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping...")
        return

    args = [
        "java", "-jar", extractor_jar_path,
        inputJar,
        jdkPath,
        input_file,
        output_file
    ]

    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"[✓] WALA-style call graph written to {output_file}")


def main():
    config_version = 'v1'
    doop_data_dir = f'/20TB/mohammad/xcorpus-total-recall/static_cgs/doop/{config_version}'
    input_programs_dir = f'{doop_data_dir}/out'
    output_dir = os.path.join(doop_data_dir, 'reformatted-v2')
    os.makedirs(output_dir, exist_ok=True)

    programs = [d for d in os.listdir(input_programs_dir) if os.path.isdir(os.path.join(input_programs_dir, d))]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for program_dir in programs:
            futures.append(executor.submit(process_program, input_programs_dir, output_dir, program_dir))

        for future in futures:
            future.result()


if __name__ == "__main__":
    main()