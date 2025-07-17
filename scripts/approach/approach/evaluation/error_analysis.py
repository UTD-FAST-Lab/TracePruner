import os
import pandas as pd
from collections import defaultdict

# inst_path = '/20TB/mohammad/xcorpus-total-recall/results/baseline/cgpruner/doop/v1_39_True/rf_programwise_0.45_trained_on_known_oversample_1.0.pkl'
inst_path = '/20TB/mohammad/xcorpus-total-recall/results/baseline/finetuned/doop/v1_39_True/codebert_programwise_0.5_trained_on_known.pkl'

instances = pd.read_pickle(inst_path)


program_map = defaultdict(list)
for inst in instances:
    program_map[inst.program].append(inst)


for program, insts in program_map.items():
    fns = []
    fps = []
    for inst in insts:
        if inst.is_known():
            if inst.get_label() == 1 and inst.get_predicted_label() == 0:
                fns.append(inst)
            elif inst.get_label() == 0 and inst.get_predicted_label() == 1:
                fps.append(inst)

    if fns or fps:
        print(f'Program: {program}')
        print(f'False Negatives: {len(fns)}')
        for fn in fns:
            print(f'{fn.src},{fn.offset},{fn.target}')

        print(f'False Positives: {len(fps)}')
        for fp in fps:
            print(f'{fp.src},{fp.offset},{fp.target}')
        print('-' * 40)


# for program, insts in program_map.items():
#     tns = []
#     tps = []
#     for inst in insts:
#         if inst.is_known():
#             if inst.get_label() == 1 and inst.get_predicted_label() == 1:
#                 tps.append(inst)
#             elif inst.get_label() == 0 and inst.get_predicted_label() == 0:
#                 tns.append(inst)

#     if tns or tps:
#         print(f'Program: {program}')
#         print(f'True Negatives: {len(tns)}')
#         for tn in tns:
#             print(f'{tn.src},{tn.offset},{tn.target}')

#         print(f'True Positives: {len(tps)}')
#         for tp in tps:
#             print(f'{tp.src},{tp.offset},{tp.target}')
#         print('-' * 40)
