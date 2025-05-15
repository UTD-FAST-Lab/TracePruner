import os
import networkx as nx
from collections import defaultdict, deque
from tqdm import tqdm
import pickle
import random

# Configuration
GRAPH_DIR = '/20TB/mohammad/data/cg_edge_repr'
FILTERED_GRAPH_DIR = '/20TB/mohammad/data/filtered_graphs'
MACRO_PREFIX = 'M'
macro_counter = 0  # Global counter for macro node names
SUBGRAPH_SIZE = 5  # Fixed size of the common subgraph
MIN_COVERAGE = 0.9  # Minimum percentage of graphs a pattern must appear in

os.makedirs(FILTERED_GRAPH_DIR, exist_ok=True)

def save_graph(graph, file_path):
    # Save the entire graph with all node and edge attributes
    with open(file_path, 'wb') as f:
        pickle.dump(graph, f)

def read_graph(file_path):
    G = nx.read_weighted_edgelist(file_path, create_using=nx.DiGraph, delimiter='\t')
    # Convert string weights to float
    for u, v, data in G.edges(data=True):
        if 'weight' in data:
            data['weight'] = float(data['weight'])
    return G


def extract_fixed_size_subgraphs(G, size):
    subgraphs = set()
    nodes = list(G.nodes())

    for node in nodes:
        # Use a queue for BFS-like expansion
        queue = deque([([(node, tgt)], tgt) for tgt in G.successors(node)])
        
        while queue:
            current_edges, current_node = queue.popleft()
            
            # If we've reached the desired size, add this as a subgraph
            if len(current_edges) == size:
                subgraph_edges = frozenset(current_edges)
                subgraphs.add(subgraph_edges)
                continue
            
            # Expand to all neighbors of the current node
            for next_node in G.successors(current_node):
                next_edge = (current_node, next_node)
                if next_edge not in current_edges:  # Avoid cycles
                    queue.append((current_edges + [next_edge], next_node))

    # Convert each subgraph to a NetworkX graph
    return [G.edge_subgraph(edges).copy() for edges in subgraphs]

def find_common_subgraphs(graphs):
    subgraph_counts = defaultdict(int)
    total_graphs = len(graphs)

    for graph in tqdm(graphs, desc="Finding common subgraphs"):
        seen = set()
        for subgraph in extract_fixed_size_subgraphs(graph, SUBGRAPH_SIZE):
            # Use frozenset of edge tuples to uniquely identify the subgraph
            # subgraph_id = frozenset((u, v, data['weight']) for u, v, data in subgraph.edges(data=True))
            subgraph_id = frozenset((u, v) for u, v, data in subgraph.edges(data=True))
            if subgraph_id not in seen:
                subgraph_counts[subgraph_id] += 1
                seen.add(subgraph_id)

    coverage_threshold = int(total_graphs * MIN_COVERAGE)
    common_subgraphs = {sg: cnt for sg, cnt in subgraph_counts.items() if cnt >= coverage_threshold}
    return common_subgraphs

def aggregate_weights(subgraph):
    edge_weights = defaultdict(list)

    for u, v, data in subgraph.edges(data=True):
        edge_weights[(u, v)].append(data['weight'])

    summary = {}
    for edge, weights in edge_weights.items():
        mean = sum(weights) / len(weights)
        variance = sum((x - mean) ** 2 for x in weights) / len(weights)
        summary[edge] = {'mean': mean, 'variance': variance, 'count': len(weights)}

    return summary

def replace_subgraph_with_macro(G, subgraph_edges, macro_name):


    # print_graph(G, "Before replacing subgraph with macro")
    
    # Create the actual subgraph with weights
    subgraph = nx.DiGraph()
    for u, v in subgraph_edges:
        if G.has_edge(u, v):
            weight = G[u][v]['weight']
            subgraph.add_edge(u, v, weight=weight)
    
    
    weight_summary = aggregate_weights(subgraph)

    subgraph_nodes = set(subgraph.nodes())
    incoming_edges = set()
    outgoing_edges = set()

    for node in subgraph_nodes:
        for pred in G.predecessors(node):
            if pred not in subgraph_nodes:
                weight = G[pred][node]['weight']
                incoming_edges.add((pred, macro_name, weight))

        for succ in G.successors(node):
            if succ not in subgraph_nodes:
                weight = G[node][succ]['weight']
                outgoing_edges.add((macro_name, succ, weight))

    G.remove_nodes_from(subgraph_nodes)
    G.add_node(macro_name, weight_summary=weight_summary)

    for u, v, w in incoming_edges:
        G.add_edge(u, v, weight=w)
    for u, v, w in outgoing_edges:
        G.add_edge(u, v, weight=w)

    # print_graph(G, "After replacing subgraph with macro")


# def print_graph(G, title):
#     print(f"{title}:")
#     for u, v, data in G.edges(data=True):
#         print(f"Edge from {u} to {v} with weight {data['weight']}")
#     print("\n")


def main():
    # Read all graphs
    graphs = []
    filenames = []
    i = 0
    for filename in tqdm(os.listdir(GRAPH_DIR), desc="Loading graphs"):
        if filename.endswith('.edg'):
            filenames.append(filename.split('.')[0])
            graph = read_graph(os.path.join(GRAPH_DIR, filename))
            graphs.append(graph)
            # i += 1
            # if i >= 10000:
            #     break

    # Find common subgraphs
    common_subgraphs = find_common_subgraphs(graphs)
    print(f"Found {len(common_subgraphs)} common subgraphs")

    # print some examples of the subgraphs
    for i, (subgraph_id, count) in enumerate(common_subgraphs.items()):
        if i < 5:  # Print only the first 5 examples
            print(f"Subgraph {i}: {subgraph_id} appears in {count} graphs")
        else:
            break

    # return
    # Replace subgraphs with macros
    global macro_counter
    for subgraph_id in common_subgraphs.keys():
        # # Recreate the subgraph from the frozen set of edges
        # subgraph = nx.DiGraph()
        # for u, v, w in subgraph_id:
        #     subgraph.add_edge(u, v, weight=w)

        macro_name = f"{MACRO_PREFIX}{macro_counter}"
        macro_counter += 1
        
        for graph in graphs:
            # if set(subgraph.edges()) <= set(graph.edges()):
            replace_subgraph_with_macro(graph, subgraph_id, macro_name)
    
    # Save updated graphs
    for i, graph in enumerate(graphs):
        save_graph(graph, f"{FILTERED_GRAPH_DIR}/{filenames[i]}.pickle")
        # print(f"Saved graph {i} with macros")
    print("All graphs saved with macros.")

if __name__ == "__main__":
    main()
