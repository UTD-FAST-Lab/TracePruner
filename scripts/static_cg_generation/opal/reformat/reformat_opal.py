
import os
import json

input_dir = '/20TB/mohammad/xcorpus-total-recall/static_cgs/opal/v1'
output_dir = '/20TB/mohammad/xcorpus-total-recall/static_cgs/opal/final'

version = "v1"  # Version of the static CGs

def get_method_signature(method):
    klass = method['declaringClass']
    name = method['name']
    parameters = [p for p in method['parameterTypes']]
    return_type = method['returnType']

    # remove the L prefix and the trailing ; from the class name 
    klass = klass[1:] if klass.startswith("L") else klass
    klass = klass[:-1] if klass.endswith(";") else klass

    # concatenate the parameters 
    parameters = ''.join(parameters)

    # concatenate everything
    signature = f"{klass}.{name}:({parameters}){return_type}"
    return signature


for program in os.listdir(input_dir):
    if program.endswith(".py"):
        continue

    print(f"Processing {program}...")
    program_path = os.path.join(input_dir, program)
    
    for file in os.listdir(program_path):

        cg_file = os.path.join(program_path, file)
        if not cg_file.endswith(".json"):
            continue
        

        with open(cg_file, "r") as f:
            cg = json.load(f)
        
        reachable_methods = cg['reachableMethods']


        all_edges = []

        for method in reachable_methods:
            for callsite in method['callSites']:
                for target in callsite['targets']:
                    edge = {
                        "method": get_method_signature(method['method']),
                        "target": get_method_signature(target),
                        "offset": callsite['pc'],
                    }

                    all_edges.append(edge)

        # write the edges to a file
        file = file.split(".")[0]  # Remove the .json extension
        tool_name, config_id = file.split("_")

        # file_name = f"{file.split('.')[0]}.csv"
        file_name = f"{tool_name}_{version}_{config_id}.csv"
        out_file_path = os.path.join(output_dir, program, file_name)
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        with open(out_file_path, "w") as f:
            f.write("method,offset,target\n")
            for edge in all_edges:
                f.write(f"{edge['method']},{edge['offset']},{edge['target']}\n")


    print(f"Finished processing {program}.")