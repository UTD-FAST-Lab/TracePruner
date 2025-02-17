import re
import os

def extract_edge_traces(trace_file, edges_dir):

    count = 0
    
    with open(trace_file, 'r') as f:
        lines = f.readlines()
        
    invoke_traces = {}
    current_invokes = {}
    
    for line in lines:
        line = line.strip()
        
        if "AgentLogger|visitinvoke:" in line:
            match = re.search(r'AgentLogger\|visitinvoke: (.+)', line)
            if match:
                instruction = match.group(1)
                current_invokes[instruction] = []
                invoke_traces[instruction] = []
            
        if "AgentLogger|addEdge:" in line:
            match = re.search(r'addEdge: (Node: < .*? > Context: Everywhere) (.*?) (Node: < .*? > Context: Everywhere)', line)
            if match:
                src = match.group(1)  # Captures the source node
                instruction = match.group(2)  # Captures the instruction
                target = match.group(3)  # Captures the target node
                
                if instruction in current_invokes:
                    count += 1

                    filename = f"{count}.log"
                    filepath = os.path.join(edges_dir, filename)
                    
                    current_invokes[instruction].append(line)

                    with open(filepath, 'w') as f:
                        f.write("\n".join(current_invokes[instruction]) + "\n\n")

                    current_invokes[instruction].pop()
                    
                    # current_invokes[instruction] = []  # Reset for the next edge
        
        for inst in current_invokes:
            current_invokes[inst].append(line)
    
    
    
if __name__ == "__main__":

    program_name = "VC4.txt"  
    edge_traces_dir = '/home/mohammad/projects/CallGraphPruner/data/edge-traces'    
    trace_folder = '/home/mohammad/projects/CallGraphPruner/data/cgs' 
    trace_path = os.path.join(trace_folder, program_name)
    
    base_dir = os.path.join(edge_traces_dir, program_name.split('.')[0])
    edges_dir = os.path.join(base_dir, "edges")
    os.makedirs(edges_dir, exist_ok=True)

    extract_edge_traces(trace_path, edges_dir)
