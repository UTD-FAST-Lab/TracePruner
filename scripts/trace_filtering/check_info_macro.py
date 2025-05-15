import os
import pickle


for file in os.listdir('/20TB/mohammad/data/filtered_graphs'):
    if file.endswith('.pickle'):
        with open(os.path.join('/20TB/mohammad/data/filtered_graphs', file), 'rb') as f:
            graph = pickle.load(f)
            print(f"Graph {file} has {len(graph.nodes())} nodes and {len(graph.edges())} edges.")

            for node in graph.nodes():
                if node.startswith('M'):
                    print(f"Node {node} is a macro node.")
                    
                    print(graph.nodes[node]['weight_summary'])

            break