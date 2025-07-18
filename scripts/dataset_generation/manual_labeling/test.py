import os
import pandas as pd

path1 = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'
path2 = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/static_cgs_no_exlusion'

count = 0
for program in os.listdir(path):
    # csv_path = os.path.join(path, program, 'wala0cfa_filtered_libs.csv')
    # csv_path = os.path.join(path, program, 'true_filtered_libs.csv')
    csv_path = os.path.join(path, program, 'false_filtered_libs.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        count += len(df)


print(count)