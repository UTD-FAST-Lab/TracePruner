import json
import pandas as pd


p = '/home/mohammad/projects/CallGraphPruner/data/edge-traces/cgs/urlcf7851e92b_Triploblastic_CSCD350_FinalProject_tgz-pJ8-MazeGame_MazeJ8/segments.json'
with open(p, 'r', encoding='utf-8') as f:
    mapping_data = json.load(f)


 # Convert mapping JSON to a DataFrame if it's a list of dictionaries
df = pd.DataFrame(mapping_data)

df = df.drop_duplicates(subset=['edge_name'], keep="first")

print(df)