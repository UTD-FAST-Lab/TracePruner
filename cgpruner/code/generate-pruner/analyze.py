import os
import csv
import json

# Directories
# data_folder = '/20TB/mohammad/cg_dataset/cgpruner'    #change this to environment
data_folder = '/20TB/mohammad'  
model = 'config'   #original, config_trace

STATIC_FOLDER = f'{data_folder}/models/pruner_{model}/output/static_cgs/testing_cgs'
PRUNED_FOLDER = f'{data_folder}/models/pruner_{model}/output/pruned_cgs'
DYNAMIC_FOLDER = f'{data_folder}/dataset-high-precision-callgraphs/full_callgraphs_set'
OUTPUT_FOLDER = f'{data_folder}/models/pruner_{model}/stats'
EVAL_FILE = os.path.join(OUTPUT_FOLDER, 'eval.json')

def load_cgs(program, config):
    static_cg = set()
    dynamic_cg = set()
    pruned_cg = set()

    # Paths for call graphs
    static_cg_file = os.path.join(STATIC_FOLDER, program, f'{program}_config_{config}.csv')
    dynamic_cg_file = os.path.join(DYNAMIC_FOLDER, program, 'wala0cfa.csv')
    pruned_cg_file = os.path.join(PRUNED_FOLDER, program, f'{program}_{config}.csv')

    # Check if static, dynamic, and pruned call graph files exist
    if not os.path.exists(static_cg_file) or not os.path.exists(dynamic_cg_file):
        return None, None, None

    if not os.path.exists(pruned_cg_file):
        print(f"Pruned graph not found for {program} config {config}, skipping...")
        return None, None, None

    # Helper function to parse edges
    def get_edges_from_file(filepath):
        edges = set()
        with open(filepath, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                edges.add((row['method'], row['target']))
        return edges

    # Read static, dynamic, and pruned call graphs
    static_cg = get_edges_from_file(static_cg_file)
    pruned_cg = get_edges_from_file(pruned_cg_file)

    # Load dynamic graph edges only if 'wiretap' field is '1'
    with open(dynamic_cg_file, mode='r') as dynamic_file:
        reader = csv.DictReader(dynamic_file)
        for row in reader:
            if row.get('wiretap') == '1':
                dynamic_cg.add((row['method'], row['target']))

    return static_cg, pruned_cg, dynamic_cg

def evaluate(static_cg, pruned_cg, dynamic_cg, program_name, config_name):
    def calculate_precision_recall(predicted_cg, true_cg):
        TP, FP, FN = 0, 0, 0

        for edge in predicted_cg:
            if edge in true_cg:
                TP += 1
            else:
                FP += 1

        for edge in true_cg:
            if edge not in predicted_cg:
                FN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        return precision, recall

    # Calculate precision and recall for static and pruned call graphs
    static_precision, static_recall = calculate_precision_recall(static_cg, dynamic_cg)
    pruned_precision, pruned_recall = calculate_precision_recall(pruned_cg, dynamic_cg)

    # Load existing JSON data if the file exists
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE, 'r') as infile:
            data = json.load(infile)
    else:
        data = {}

    # Create or update the program entry in the JSON structure
    if program_name not in data:
        data[program_name] = {}

    data[program_name][config_name] = {
        'original(static)': {'precision': static_precision, 'recall': static_recall},
        'pruned': {'precision': pruned_precision, 'recall': pruned_recall}
    }

    # Write the updated data back to the JSON file
    with open(EVAL_FILE, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print(f"Evaluation completed for {program_name}, {config_name}:")
    print(f"  Original (Static): Precision = {static_precision}, Recall = {static_recall}")
    print(f"  Pruned: Precision = {pruned_precision}, Recall = {pruned_recall}")

def overall_evaluation():
    '''Read the eval file and calculate the cumulative precision/recall of pruned and original cgs for each config.'''
    
    # Initialize an overall dictionary to store cumulative precision and recall for each config
    overall = {
        f"config_{i}": {"original(static)": {"precision": 0, "recall": 0}, "pruned": {"precision": 0, "recall": 0}, "count": 0}
        for i in range(11)  # Assuming config ranges from 0 to 10
    }

    # Read the eval file
    with open(EVAL_FILE, 'r') as eval_file:
        data = json.load(eval_file)

    # Iterate through each program and its configurations
    for program_name, configs in data.items():
        for config_name, results in configs.items():
            config_key = config_name  # e.g., "config_1"
            
            # Accumulate precision and recall for the original static call graph
            overall[config_key]["original(static)"]["precision"] += results["original(static)"]["precision"]
            overall[config_key]["original(static)"]["recall"] += results["original(static)"]["recall"]

            # Accumulate precision and recall for the pruned call graph
            overall[config_key]["pruned"]["precision"] += results["pruned"]["precision"]
            overall[config_key]["pruned"]["recall"] += results["pruned"]["recall"]

            # Increment the count for averaging later
            overall[config_key]["count"] += 1

    # Calculate average precision and recall for each config
    for config_key, metrics in overall.items():
        count = metrics["count"]
        if count > 0:
            # Average the precision and recall for original and pruned graphs
            metrics["original(static)"]["precision"] /= count
            metrics["original(static)"]["recall"] /= count
            metrics["pruned"]["precision"] /= count
            metrics["pruned"]["recall"] /= count

    # Remove the count field as it's no longer needed
    for config_key in overall.keys():
        overall[config_key].pop("count")

    # Output the overall evaluation
    overall_output_file = os.path.join(OUTPUT_FOLDER, 'overall_eval.json')
    with open(overall_output_file, 'w') as outfile:
        json.dump(overall, outfile, indent=4)

    print(f"Overall evaluation saved to {overall_output_file}")


def main():
    programs = os.listdir(STATIC_FOLDER)

    for program in programs:
        for config in range(11):  # Assuming config ranges from 0 to 10
            static_cg, pruned_cg, dynamic_cg = load_cgs(program, config)
            if static_cg is None or pruned_cg is None or dynamic_cg is None:
                continue
            evaluate(static_cg, pruned_cg, dynamic_cg, program, f"config_{config}")
    
    overall_evaluation()

if __name__ == '__main__':
    main()
