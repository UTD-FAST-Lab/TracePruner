import os

output_dir = "/20TB/mohammad/xcorpus-total-recall/static_cgs/doop/final"
input_dir = "/20TB/mohammad/xcorpus-total-recall/static_cgs/doop"

for version in os.listdir(input_dir):
    if version.startswith("v"):
        print(f"Processing version: {version}")
        version_dir = os.path.join(input_dir, version, "reformatted-v2")
        for program_config in os.listdir(version_dir):
            program, config_id = program_config.split('_')
            program_config_dir = os.path.join(version_dir, program_config)
            input_file_path = os.path.join(program_config_dir, 'CallGraphEdge.csv')


            output_name = f"doop_{version}_{config_id}.csv"
            file_output_dir = os.path.join(output_dir, program)
            file_output_path = os.path.join(file_output_dir, output_name)
            if not os.path.exists(file_output_path):
                if os.path.exists(input_file_path):
                    os.makedirs(file_output_dir, exist_ok=True)
                    print(f"Copying {input_file_path} to {file_output_path}")
                    with open(input_file_path, 'r') as infile, open(file_output_path, 'w') as outfile:
                        outfile.write(infile.read())
                else:
                    print(f"Input file {input_file_path} does not exist. Skipping.")
            else:
                print(f"Output file {file_output_path} already exists. Skipping.")
                