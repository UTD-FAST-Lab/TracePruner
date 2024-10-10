import os
import concurrent.futures

EXTRA_INFO_SCRIPT = 'tools/balancer-cg/generate_extra_information_from_dataset.py'
INFERENCE_SCRIPT = 'tools/balancer-cg/inference.py'
TMP_FOLDER = "tmp"

data_folder = '/20TB/mohammad'    #change this to environment
# data_folder = '/home/mohammad/projects/CallGraphPruner_data'    #change this to environment
model_type = 'config'     #original, cofig, config_trace

CG_DIR = f'{data_folder}/models/pruner_{model_type}/output/static_cgs/testing_cgs'
OUTPUT_DIR = f'{data_folder}/models/pruner_{model_type}/output/pruned_cgs'

# CLASSIFIER_PATH = f'{data_folder}/dataset-high-precision-callgraphs/learned_classifiers/wala.joblib'
CLASSIFIER_PATH = f'{data_folder}/models/pruner_{model_type}/pruner_{model_type}.pkl'
CUTOFF = 0.45


# Function to process each program and config
def run_model(INPUT_FILE, program, config):
    EXTRA_INFO_OUT_FILE = f'tmp/callgraph_extra_feat_{program}_{config}.csv'
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, program, f'{program}_{config}.csv')
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    # Generate extra information
    os.system(f'python3 {EXTRA_INFO_SCRIPT} {INPUT_FILE} {EXTRA_INFO_OUT_FILE}')

    # Run inference with cutoff value
    os.system(f'python3 {INFERENCE_SCRIPT} {EXTRA_INFO_OUT_FILE} {CLASSIFIER_PATH} {CUTOFF} > {OUTPUT_FILE}')


def main():
    programs = [p for p in os.listdir(CG_DIR) if os.path.isdir(os.path.join(CG_DIR, p))]
    # print(programs)
    ignore_programs = (
        'url1a41b8e9f2_mr1azl_INF345_tgz-pJ8-TestParserJ8',
        'url27a7ae3508_serkan_ozal_ocean_of_memories_tgz-pJ8-com_zeroturnaround_rebellabs_oceanofmemories_article1_objectlayout_ObjectMemoryLayoutDemoJ8'
    )
    programs = [p for p in programs if p not in ignore_programs] 
    # print(programs)

    # Use ThreadPoolExecutor with 10 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for program in programs:
            program_config_files = [file for file in os.listdir(os.path.join(CG_DIR, program)) if file.endswith('.csv')]
            for file in program_config_files:
                config = file.replace('.csv', '').split('_')[-1]
                INPUT_FILE = os.path.join(CG_DIR, program, file)
                
                # Submit the task to the thread pool
                futures.append(executor.submit(run_model, INPUT_FILE, program, config))

        # Wait for all threads to finish
        concurrent.futures.wait(futures)


if __name__ == '__main__':
    main()
