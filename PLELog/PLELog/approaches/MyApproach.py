import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
from sklearn.decomposition import FastICA
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from representations.sequences.statistics import Sequential_TF
from preprocessing.datacutter.SimpleCutting import cut_by_613
from preprocessing.AutoLabeling import Probabilistic_Labeling
from preprocessing.Preprocess import Preprocessor
from module.Optimizer import Optimizer
from module.Common import data_iter, generate_tinsts_binary_label, batch_variable_inst
from models.gru import AttGRUModel
from utils.Vocab import Vocab

from entities.instances import Instance

import pandas as pd
import json
import networkx as nx
from collections import defaultdict
import os
import pickle

# from graphs import embed_graphs_node2vec, embed_graphs_graph2vec
from graphs import embed_graphs_node2vec_original



def load_hash_map(path):
    """Load existing hash map from file, or create a new one."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["string_to_id"], {int(k): v for k, v in data["id_to_string"].items()}
    print("not exist: ", path)
    return {}, {}


def save_hash_map(id2log_path, string_to_id, id_to_string):
        """Save hash map to a JSON file."""
        with open(id2log_path, "w", encoding="utf-8") as f:
            json.dump({"string_to_id": string_to_id, "id_to_string": id_to_string}, f, indent=4)


def gen_id2embd(id2log_path, template):
    '''loads the id2log dictionary and for each id generate the embedding and write to a file'''
    string_to_id, id_to_string = load_hash_map(id2log_path)

    id2embd = template.present(id_to_string)

    return id2embd, string_to_id, id_to_string
    

def parse_cg_trace(id2log_path, trace, string_to_id, id_to_string):
    '''encodes each trace(edge) to its method ids e.g., 1,1 1,4 5,3 3,6 + the var info logs and write to a file'''
    
    unique_strings = set(string_to_id.keys())  # Get current unique strings
    changed = False

    path = ''
    encoded_trace = {}
    # with open(path, 'w') as file:
    for line_num, line in trace.items():
        if line.startswith("AgentLogger|CG_edge: "):
            # Remove the prefix before splitting
            line = line[len("AgentLogger|CG_edge: "):]

            # Split by "->" with optional spaces
            parts = re.split(r"\s*->\s*", line)
            if len(parts) == 2:
                left, right = parts

                # Assign new IDs if needed
                for s in [left, right]:
                    if s not in unique_strings:
                        changed = True
                        new_id = len(string_to_id) + 1
                        string_to_id[s] = new_id
                        id_to_string[new_id] = s
                        unique_strings.add(s)

                encoded_trace[line_num] = f"{string_to_id[left]},{string_to_id[right]}"
        else:
            encoded_trace[line_num] = line

    if changed:
        save_hash_map(id2log_path, string_to_id, id_to_string)


    return encoded_trace


def parse_br_trace(id2log_path, trace, string_to_id, id_to_string):

    unique_strings = set(string_to_id.keys())  # Get current unique strings
    changed = False

    changed = False
    encoded_trace = {}
    # with open(path, 'w') as file:
    for line_num, line in trace.items():
        if line.startswith("AgentLogger|BRANCH: "):
            # Remove the prefix before matching
            line = line[len("AgentLogger|BRANCH: "):]

            # Use a regex to capture the method signature, branch type, and branch ID
            m = re.match(r"^(.*):([A-Z]+)#(\d+)$", line)
            if m:
                method, branch_type, branch_id = m.group(1), m.group(2), m.group(3)
                
                # Assign new IDs if needed
                if method not in unique_strings:
                    changed = True
                    new_id = len(string_to_id) + 1
                    string_to_id[method] = new_id
                    id_to_string[new_id] = method
                    unique_strings.add(method)
                
                # encode branch type (0 -> else , 1-> if)
                if branch_type == "IF":
                    branch_type = 1
                elif branch_type == "ELSE":
                    branch_type = 0
                else:
                    raise ValueError("branch type is not defined!")

                # Save encoded results
                encoded_trace[line_num] = f"{string_to_id[method]},{branch_id},{branch_type}\n"

            else:
                encoded_trace[line_num] = line

    if changed:
        save_hash_map(id2log_path, string_to_id, id_to_string)

    return encoded_trace



def seperate_method_brgraphs(encoded_trace):
    '''makes the trace for each method_id'''

    method2trace = defaultdict(list)

    for line_num, line in encoded_trace.items():
        if '_INFO' in line:
            continue

        elif 'AgentLogger' in line:
            continue
        else:
            method_id, id, label = line.split(',')
            method2trace[method_id].append((id, label))

    return method2trace

def create_br_graphs(encoded_trace):
    '''create the graph representation of the branches in the following format: {node_id: graph}'''
    '''(method_id, branch_id, if,else) -> (1,2,0)'''

    graphs = {}

    method2trace = seperate_method_brgraphs(encoded_trace)   #{method_id: [(id, label)]}

    for method_id, lines in method2trace.items():
        graph = nx.DiGraph()
        nodes = set()
        edges = defaultdict(int)
        for i in range(len(lines) - 1):
            
            src_branch_id, src_label = lines[i]
            src_branch_id = int(src_branch_id)
            src_label = int(src_label)
            
            trg_branch_id, trg_label = lines[i+1]
            trg_branch_id = int(trg_branch_id)
            trg_label = int(trg_label)

            nodes.add((src_branch_id, src_label))
            nodes.add((trg_branch_id, trg_label))

            edge_key = ((src_branch_id, src_label), (trg_branch_id, trg_label))
            edges[edge_key] += 1

        for node_id in nodes:
            graph.add_node(node_id)


        for key, freq in edges.items():
            graph.add_edge(key[0], key[1], weight=freq)


        graphs[method_id] = graph
    
    return graphs


def create_cg_graph(encoded_trace, brgraphs):
    """create the graph representation of the trace"""

    G = nx.DiGraph()
    nodes = set()
    edges = defaultdict(int)
    info_lines = []
    start_node = None
    end_node = None

    for line_num, line in encoded_trace.items():
        if '_INFO' in line:
            info_lines.append(line)

        # elif 'AgentLogger' in line:
        elif 'AgentLogger' in line:
            continue
        else:
            try:
                src, trg = line.split(',')
            except ValueError:
                raise ValueError(line)
            src = int(src)
            trg = int(trg)

            # Track the first src as start_node
            if start_node is None:
                start_node = src

            # Always update end_node with the latest trg
            end_node = trg

            nodes.add(src)
            nodes.add(trg)

            edges[(src,trg)] += 1


    for node_id in nodes:
        if node_id == start_node:
            # Extract the first two info lines (visitInvoke format)
            pointer_info_line, cg_info_line = info_lines[:2]
        elif node_id == end_node:
            # Extract the last two info lines (addEdge format)
            pointer_info_line, cg_info_line = info_lines[-2:]
        else:
            pointer_info_line, cg_info_line = None, None

        # Process and parse pointer info
        pointer_info = None
        is_start = False
        if pointer_info_line:
            if "AgentLogger|POINTER_INFO(visitInvoke):" in pointer_info_line:
                is_start = True
                json_start = pointer_info_line.find("{")
                if json_start != -1:
                    try:
                        pointer_info = json.loads(pointer_info_line[json_start:])
                    except json.JSONDecodeError:
                        print(f"Error parsing POINTER_INFO for node {node_id}: {pointer_info_line}")
            elif "AgentLogger|POINTER_INFO(addEdge):" in pointer_info_line:
                json_start = pointer_info_line.find("{")
                if json_start != -1:
                    try:
                        pointer_info = json.loads(pointer_info_line[json_start:])
                    except json.JSONDecodeError:
                        print(f"Error parsing POINTER_INFO (addEdge) for node {node_id}: {pointer_info_line}")

        # Process and parse call graph info
        call_graph_info = None
        if cg_info_line:
            if "AgentLogger|CALL_GRAPH_INFO(visitInvoke):" in cg_info_line:
                json_start = cg_info_line.find("{")
                if json_start != -1:
                    try:
                        call_graph_info = json.loads(cg_info_line[json_start:])
                    except json.JSONDecodeError:
                        print(f"Error parsing CALL_GRAPH_INFO for node {node_id}: {cg_info_line}")
            elif "AgentLogger|CALL_GRAPH_INFO(addEdge):" in cg_info_line:
                json_start = cg_info_line.find("{")
                if json_start != -1:
                    try:
                        call_graph_info = json.loads(cg_info_line[json_start:])
                    except json.JSONDecodeError:
                        print(f"Error parsing CALL_GRAPH_INFO (addEdge) for node {node_id}: {cg_info_line}")

        # Store the extracted information
        if pointer_info or call_graph_info:
            var_info = {"entrypoint": is_start, "pointer_info": pointer_info, "call_graph_info": call_graph_info}
        else:
            var_info = None  # No relevant info for non-start/non-end nodes

        # Add node with attributes
        G.add_node(node_id, embedding=id2embd[node_id], brgraph=brgraphs.get(node_id), info=var_info)


    for key, freq in edges.items():
        G.add_edge(key[0], key[1], weight=freq)

    return G





# def embedd_trace(id2embd, encoded_trace):
#     '''for each encoded trace, calculate the embedding of the whole trace according to the id2embd of the templates and embedding of the var info logs'''
#     '''
#     - loop through the semi encoded trace for each line
#     - give weight 2 to the src and 1 to target ( to differentiate the e.g., 12 and 21) and aggegate them to get the embedding of a single template log.
#     - if the line is var info, then instead of using id2embed call the embedding method
#     - then after all of the lines has their corresponding encodding, then do the weighted aggregation with dymamic weights to var info.
#     - return the embedding of the trace.
#     '''
#     normal_embed_logs = []
#     imp_raw_logs = {}

#     for line_num, line in encoded_trace.items():

#         if '-INFO' in line:
#             # continue
#             imp_raw_logs[line_num] = line

#         elif 'visitInvoke: ' in line or 'addEdge: ' in line or 'AgentLogger' in line:
#             continue
#         else:
#             src, trg = line.split(',')
#             src = int(src)
#             trg = int(trg)
#             src_embd = id2embd[src]
#             trg_embd = id2embd[trg]

#             aggregated_log = 2 * np.array(src_embd) + np.array(trg_embd)
#             normal_embed_logs.append(aggregated_log)

#     imp_embed_logs = template.present(imp_raw_logs)
#     imp_embed_logs = list(imp_embed_logs.values())
#     # imp_embed_logs = list()

#     n_imp = len(imp_embed_logs)
#     n_total = len(normal_embed_logs) + len(imp_embed_logs)

#     # Avoid division by zero if no important sentences
#     if n_imp == 0:
#         w_imp = 1.0  # Default to equal weight
#     else:
#         w_imp = 1 + (n_total / n_imp)  # Adaptive weight for important sentences
#     w_non_imp = 1.0  # Default weight for non-important sentences

#     total_weight = w_non_imp * len(normal_embed_logs) + w_imp * len(imp_embed_logs)

    
#     for i in range(len(imp_embed_logs)):
#         imp_embed_logs[i] = np.array(imp_embed_logs[i]) * (w_imp/total_weight)
    
#     for i in range(len(normal_embed_logs)):
#         normal_embed_logs[i] = np.array(normal_embed_logs[i]) * (w_non_imp/total_weight)
    
#     aggregated_embed_logs = []
#     aggregated_embed_logs.extend(normal_embed_logs)
#     aggregated_embed_logs.extend(imp_embed_logs)


#     final_embedding = np.sum(aggregated_embed_logs, axis=0)

#     return final_embedding



def load_trace(path):
    '''Load the trace of a single edge and return a dictionary {p_id_line_number: line}.'''
    
    with open(path, 'r') as file:
        lines = {i + 1: line.strip() for i, line in enumerate(file.readlines())}
    
    return lines


def load_label(program_path):
    """loads the labels of the program"""

    csv_path = os.path.join(program_path, "output.csv")
    df = pd.read_csv(csv_path, dtype={"edge_name": str})
    return df


# def embedding_trace(seq):
#     """embedd the trace of a single edge"""

#     id_temps = template.present(seq)

#     var_info_ids = []
#     for id, log in seq.items():
#         '''add weights to var info logs in a sequence'''

#         if "-INFO" in log:
#             var_info_ids.append(id)

#     # Number of important sentences and total sentences
#     n_imp = len(var_info_ids)
#     n_total = len(seq)

#     # Avoid division by zero if no important sentences
#     if n_imp == 0:
#         w_imp = 1.0  # Default to equal weight
#     else:
#         w_imp = 1 + (n_total / n_imp)  # Adaptive weight for important sentences
#     w_non_imp = 1.0  # Default weight for non-important sentences

#     # Assign weights dynamically
#     weights = {id: w_imp if id in var_info_ids else w_non_imp for id in id_temps}

#     # Normalize weights
#     total_weight = sum(weights.values())
#     normalized_weights = {id: w / total_weight for id, w in weights.items()}

#     # Compute weighted sequence embedding
#     final_embedding = sum(normalized_weights[id] * np.array(id_temps[id]) for id in id_temps)

#     return final_embedding


def create_instance(p_id, edge_name, seq, embedding, label):
    """create a single instance for each edge"""

    if str(label) == '1':
        label = 'Normal'
    elif str(label) == '0':
        label = "Anomalous"

    # new_instance = Instance(block_id=f'{p_id}_{edge_name}', log_sequence=seq.values(), label=label)
    new_instance = Instance(block_id=f'{p_id}_{edge_name}', log_sequence=seq, label=label)
    new_instance.repr = embedding
    return new_instance


def split_631(instances):
    """split to train dev and test 6,1,3"""
    return cut_by_613(instances)


def feature_reduction(instances):
    """feature reduction for clustering"""


    embeddins = []
    for inst in instances:
        embeddins.append(inst.repr)

    embddings = np.asarray(embeddins, dtype=np.float64)


    n_components = min(50, embddings.shape[0])
    print(f"Start FastICA, target dimension: {n_components}")

    # Apply FastICA
    transformer = FastICA(n_components=n_components)
    train_reprs = transformer.fit_transform(embddings)

    # print("Reduced:", train_reprs)

    for i, inst in enumerate(instances):
        inst.repr = train_reprs[i]


def probability_labeling(train):
    '''clustering algorithm'''

    prob_label_res_file = os.path.join(
                                       'results_n2v/PLELog/' + "testdata" + '_' +
                                       '/prob_label_res/mcs-' + str(100) + '_ms-' + str(100))
    rand_state = os.path.join(
                              'results_n2v/PLELog/' + "testdata" + '_' +
                              '/prob_label_res/random_state')


    label_generator = Probabilistic_Labeling(min_samples=50, min_clust_size=75,
                                             res_file=prob_label_res_file, rand_state_file=rand_state)    

    
    # Probabilistic labeling.
    # Sample normal instances.
    train_normal = [x for x, inst in enumerate(train) if inst.label == 'Normal']
    normal_ids = train_normal[:int(0.5 * len(train_normal))]
    

    labeled_train = label_generator.auto_label(train, normal_ids)

    evaluate_cluster(labeled_train, normal_ids)

    return labeled_train


def evaluate_cluster(labeled_train, normal_ids):
    '''evaluate the effectiveness of clustering, needs work!!'''

    # Below is used to test if the loaded result match the original clustering result.
    TP, TN, FP, FN = 0, 0, 0, 0

    for inst in labeled_train:
        if inst.predicted == 'Normal':
            if inst.label == 'Normal':
                TP += 1
            else:
                FP += 1
        else:
            if inst.label == 'Anomalous':
                TN += 1
            else:
                FN += 1
    from utils.common import get_precision_recall

    print(len(normal_ids))
    print('TP %d TN %d FP %d FN %d' % (TP, TN, FP, FN))
    p, r, f = get_precision_recall(TP, TN, FP, FN)
    print('%.4f, %.4f, %.4f' % (p, r, f))








def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    


if __name__ == '__main__':


    programs_path = "/app/data/edge-traces/cgs"
    branches_path = "/app/data/edge-traces/branches"
    id2log_path = '/app/data/WALA_hash_map.json'

    graphs_path = '/app/data/cached_graphs.pkl'
    embeddings_path = '/app/data/cached_graph_embeddings.pkl'

    reduction = False
    instances = []


    # Load or generate graphs

    
    if os.path.exists(graphs_path):
        # print("Loading cached graphs...")
        graphs = load_pickle(graphs_path)
    else:

        # Embed WALA method names
        template = Simple_template_TF_IDF()
        id2embd, string_to_id, id_to_string = gen_id2embd(id2log_path, template)

        graphs = {}
        for p_id, program in enumerate(os.listdir(programs_path)):
            program_path = os.path.join(programs_path, program)
            labels_df = load_label(program_path)
            edges_path = os.path.join(programs_path, program, "edges")

            for edge in os.listdir(edges_path):
                edge_path_cg = os.path.join(edges_path, edge)
                edge_path_br = os.path.join(branches_path, program, "edges", edge)

                edge_id = edge.split('.')[0]
                label_row = labels_df.loc[labels_df["edge_name"] == edge_id, "wiretap"]

                if label_row.empty:
                    continue

                label = label_row.values[0]

                cg_trace = load_trace(edge_path_cg)
                br_trace = load_trace(edge_path_br)
                print("Traces loaded for:", program, edge)

                cg_encoded_trace = parse_cg_trace(id2log_path, cg_trace, string_to_id, id_to_string)
                br_encoded_trace = parse_br_trace(id2log_path, br_trace, string_to_id, id_to_string)

                brgraphs = create_br_graphs(br_encoded_trace)
                graph = create_cg_graph(cg_encoded_trace, brgraphs)

                print(f"Graph {program}-{edge} has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

                graphs[(program, edge, label)] = graph

        save_pickle(graphs, graphs_path)
        print("Graphs saved.")


    # first_key = list(graphs.keys())[0]
    # first_graph = graphs[first_key]

    # # Get first edge (u, v)
    # u, v = list(first_graph.edges())[0]

    # # Modify the weight
    # if 'weight' in first_graph[u][v]:
    #     print("Old weight:", first_graph[u][v]['weight'])

    # first_graph[u][v]['weight'] = 5  # example new weight
    # print("New weight set.", first_graph[u][v]['weight'])

    # Load or generate embeddings
    if os.path.exists(embeddings_path):
        # print("Loading cached graph embeddings...")
        g_embeddings_graph2vec = load_pickle(embeddings_path)
    else:
        # g_embeddings_graph2vec = embed_graphs_graph2vec(list(graphs.values()))
        g_embeddings_graph2vec = embed_graphs_node2vec_original(list(graphs.values()))
        save_pickle(g_embeddings_graph2vec, embeddings_path)
        print("Graph embeddings saved.")


    # Create instances
    for i, (key, graph) in enumerate(graphs.items()):
        program, edge, label = key
        embedding = g_embeddings_graph2vec[i]
        inst = create_instance(program, edge, None, embedding, label)
        instances.append(inst)

    # Split and reduce
    train, dev, test = split_631(instances)

    if reduction:
        feature_reduction(train)


    n_false = 0
    n_true = 0
    i = 0
    print('program,','edge,','label')
    for inst in train:
        if inst.label == 'Anomalous':
            id = inst.id.split('_')
            edge = id[-1]
            program = '_'.join(id[:-1])
            print(program.strip(),',',edge.strip(),',')
            n_false += 1
        else:
            n_true += 1


    # labeled_train = probability_labeling(train)

    # # Debug print
    # for inst in labeled_train:
    #     print(inst.id)
    #     print("predicted: ", inst.predicted)
    #     print("label: ", inst.label)
    #     print(inst.confidence)
    #     print("------------------------------------------")


