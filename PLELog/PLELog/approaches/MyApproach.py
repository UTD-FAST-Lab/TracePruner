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

def load_hash_map(path):
    """Load existing hash map from file, or create a new one."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["string_to_id"], {int(k): v for k, v in data["id_to_string"].items()}
    return {}, {}


def gen_id2embd():
    '''loads the id2log dictionary and for each id generate the embedding and write to a file'''
    id2log_path = ''
    string_to_id, id_to_string = load_hash_map(id2log_path)

    id2embd = template.present(id_to_string)

    return id2embd, string_to_id
    

def parse_trace(trace, string_to_id):
    '''encodes each trace(edge) to its method ids e.g., 1,1 1,4 5,3 3,6 + the var info logs and write to a file'''
    
    path = ''
    encoded_trace = {}
    with open(path, 'w') as file:
        for line_num, line in trace.items():
            if line.startswith("AgentLogger|CG_edge: "):
                # Remove the prefix before splitting
                line = line[len("AgentLogger|CG_edge: "):]

                # Split by "->" with optional spaces
                parts = re.split(r"\s*->\s*", line)
                if len(parts) == 2:
                    left, right = parts
                    encoded_trace[line_num] = f"{string_to_id[left]},{string_to_id[right]}"
                    file.write(f"{string_to_id[left]},{string_to_id[right]}\n")
            else:
                file.write(line)
                encoded_trace[line_num] = line

    return encoded_trace



def embedd_trace(id2embd, encoded_trace):
    '''for each encoded trace, calculate the embedding of the whole trace according to the id2embd of the templates and embedding of the var info logs'''
    '''
    - loop through the semi encoded trace for each line
    - give weight 2 to the src and 1 to target ( to differentiate the e.g., 12 and 21) and aggegate them to get the embedding of a single template log.
    - if the line is var info, then instead of using id2embed call the embedding method
    - then after all of the lines has their corresponding encodding, then do the weighted aggregation with dymamic weights to var info.
    - return the embedding of the trace.
    '''
    normal_embed_logs = []
    imp_raw_logs = {}

    for line_num, line in encoded_trace:
        if '-INFO' in line:
            imp_raw_logs[line_num] = line

        else:
            src, trg = line.split(',')
            src_embd = id2embd[src]
            trg_embd = id2embd[trg]

            aggregated_log = 2 * np.array(src_embd) + np.array(trg_embd)
            normal_embed_logs.append(aggregated_log)

    imp_embed_logs = template.present(imp_raw_logs)
    imp_embed_logs = imp_embed_logs.values()

    n_imp = len(imp_embed_logs)
    n_total = len(normal_embed_logs) + len(imp_embed_logs)

    # Avoid division by zero if no important sentences
    if n_imp == 0:
        w_imp = 1.0  # Default to equal weight
    else:
        w_imp = 1 + (n_total / n_imp)  # Adaptive weight for important sentences
    w_non_imp = 1.0  # Default weight for non-important sentences

    total_weight = sum(w_non_imp * len(normal_embed_logs) + w_imp * len(imp_embed_logs))
    
    for i in range(len(imp_embed_logs)):
        imp_embed_logs[i] = np.array(imp_embed_logs[i]) * (w_imp/total_weight)
    
    for i in range(len(normal_embed_logs)):
        normal_embed_logs[i] = np.array(normal_embed_logs[i]) * (w_non_imp/total_weight)

    
    aggregated_embed_logs = normal_embed_logs.append(imp_embed_logs)

    final_embedding = sum(log for log in aggregated_embed_logs)

    return final_embedding



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

    new_instance = Instance(block_id=f'{p_id}_{edge_name}', log_sequence=seq.values(), label=label)
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
                                       'results/PLELog/' + "testdata" + '_' +
                                       '/prob_label_res/mcs-' + str(100) + '_ms-' + str(100))
    rand_state = os.path.join(
                              'results/PLELog/' + "testdata" + '_' +
                              '/prob_label_res/random_state')


    label_generator = Probabilistic_Labeling(min_samples=100, min_clust_size=100,
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
                TN += 1
            else:
                FN += 1
        else:
            if inst.label == 'Anomalous':
                TP += 1
            else:
                FP += 1
    from utils.common import get_precision_recall

    print(len(normal_ids))
    print('TP %d TN %d FP %d FN %d' % (TP, TN, FP, FN))
    p, r, f = get_precision_recall(TP, TN, FP, FN)
    print('%.4f, %.4f, %.4f' % (p, r, f))





if __name__ == '__main__':

    template =  Simple_template_TF_IDF()
    programs_path = "/app/data/edge-traces/cgs"
    instances = []
    reduction = False

    

    for p_id, program in enumerate(os.listdir(programs_path)):
        program_path = os.path.join(programs_path, program)
        labels_df = load_label(program_path)
        edges_path = os.path.join(programs_path, program, "edges")
        n_normal = 0
        n_anormal = 0
        for edge in os.listdir(edges_path):
            edge_path = os.path.join(edges_path, edge)
            edge_id = edge.split('.')[0]
            label = labels_df.loc[labels_df["edge_name"] == edge_id, "wiretap"].values
            if len(label) <= 0:
                continue
            label = label[0]
            if str(label) == '1':
                n_normal += 1
            elif str(label) == "0":
                n_anormal += 1

            seq = load_trace(edge_path)
            print('trace loaded ....')

            embd = embedding_trace(seq)
            print('embedding done ...')

            # new instance
            inst = create_instance(p_id, edge, seq, embd, label)

            print("instance created ...")

            instances.append(inst)

            if n_normal > 2 and n_anormal > 2:
                break

    
    train, dev, test = split_631(instances)

    if reduction:
        reduction(train)

    labeled_train = probability_labeling(train)


