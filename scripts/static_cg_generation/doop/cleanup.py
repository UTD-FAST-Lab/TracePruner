import os
import shutil

out_dir = "/20TB/mohammad/xcorpus-total-recall/static_cgs/doop/out"  # Replace with your actual path

for program in os.listdir(out_dir):
    program_path = os.path.join(out_dir, program)
    if not os.path.isdir(program_path):
        continue

    callgraph_path = os.path.join(program_path, 'database', "CallGraphEdge.csv")
    
    # Temporarily move CallGraph.csv (if exists)
    temp_path = None
    if os.path.isfile(callgraph_path):
        temp_path = os.path.join(out_dir, f"{program}_CallGraph.csv")
        shutil.copy2(callgraph_path, temp_path)

    # Delete the whole program folder
    shutil.rmtree(program_path)

    # Recreate folder and move CallGraph.csv back in
    if temp_path:
        os.makedirs(program_path, exist_ok=True)
        shutil.move(temp_path, os.path.join(program_path, "CallGraphEdge.csv"))
