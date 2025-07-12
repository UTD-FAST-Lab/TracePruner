import os
import pandas as pd

s_cgs_path = '/20TB/mohammad/my-dataset/no_libs'
d_cgs_path = '/20TB/mohammad/njr_results/dynamic_filtered_libs'

instances_path = '/20TB/mohammad/njr_results/results_new/baseline/finetune/nn_raw_0.5.pkl'

d_dfs = []
for program in os.listdir(s_cgs_path):
    d_path = os.path.join(d_cgs_path, program, 'dynamic_filtered_libs.csv')
    if os.path.exists(d_path):
        d_df = pd.read_csv(d_path)
        d_df['program'] = program
        d_dfs.append(d_df)

if d_dfs:
    combined_d_df = pd.concat(d_dfs, ignore_index=True)


# read the instances file
instances = pd.read_pickle(instances_path)

# creating static cg set
s_set = set()
for inst in instances:
    if inst.get_predicted_label() == 1:
        s_set.add((inst.src, inst.offset, inst.target))

# create a set of all method, offset, target for both dataframes
d_set = set(zip(combined_d_df['method'], combined_d_df['offset'], combined_d_df['target']))

# calculate the precision = len(s_set intersection d_set) / len(s_set) if s_set else 0
precision = len(s_set.intersection(d_set)) / len(s_set) if s_set else 0
# calculate the recall = len(s_set intersection d_set) / len(d_set) if d_set else 0
recall = len(s_set.intersection(d_set)) / len(d_set) if d_set else 0

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')



# import os
# import pandas as pd

# s_cgs_path = '/20TB/mohammad/my-dataset/no_libs'
# d_cgs_path = '/20TB/mohammad/njr_results/dynamic_filtered_libs'

# instances_path = '/20TB/mohammad/njr_results/results_new/baseline/finetune/nn_0.5_trained_on_known.pkl'

# d_dfs = []
# for program in os.listdir(s_cgs_path):
#     d_path = os.path.join(d_cgs_path, program, 'dynamic_filtered_libs.csv')
#     if os.path.exists(d_path):
#         d_df = pd.read_csv(d_path)
#         d_df['program'] = program
#         d_dfs.append(d_df)

# if d_dfs:
#     combined_d_df = pd.concat(d_dfs, ignore_index=True)


# # read the instances file
# instances = pd.read_pickle(instances_path)

# # creating static cg set
# s_set = set()
# for inst in instances:
#     if inst.is_known() or inst.ground_truth is not None:
#         if inst.get_predicted_label() == 1:
#             s_set.add((inst.src, inst.offset, inst.target))


# final_path = '/20TB/mohammad/njr_results/final_manual.csv'
# final_df = pd.read_csv(final_path)

# # add the final.csv file where label is 1 to d_dfs
# d_dfs.append(final_df[final_df['label'] == 1])

# # create a set of all method, offset, target for both dataframes
# d_set = set(zip(combined_d_df['method'], combined_d_df['offset'], combined_d_df['target']))

# # calculate the precision = len(s_set intersection d_set) / len(s_set) if s_set else 0
# precision = len(s_set.intersection(d_set)) / len(s_set) if s_set else 0
# # calculate the recall = len(s_set intersection d_set) / len(d_set) if d_set else 0
# recall = len(s_set.intersection(d_set)) / len(d_set) if d_set else 0

# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')