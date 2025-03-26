import json
import pandas as pd


p = '/home/mohammad/projects/CallGraphPruner/data/edge-traces/cgs/url1a41b8e9f2_mr1azl_INF345_tgz-pJ8-TestParserJ8/segments.json'
with open(p, 'r', encoding='utf-8') as f:
    mapping_data = json.load(f)


 # Convert mapping JSON to a DataFrame if it's a list of dictionaries
df = pd.DataFrame(mapping_data)

df = df.drop_duplicates(subset=['edge_name'], keep="first")

print(df)