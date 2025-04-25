import os
import pandas as pd

trace_edges = '/home/mohammad/projects/CallGraphPruner/data/edge-traces/new_cgs'
static_edges = '/home/mohammad/projects/CallGraphPruner/data/static-cgs/wala_1.5.9'

program = 'url0e7d57473a_kyorohiro_HetimaUtil_tgz-pJ8-net_hetimatan_net_http_HttpServer3xxJ8'

t_df_path = os.path.join(trace_edges, program , 'edges.csv')
s_df_path = os.path.join(static_edges, program , 'wala0cfa_filtered.csv')

# Read dynamic trace with manual column names
t_df = pd.read_csv(t_df_path, header=None, names=['edge_name', 'method', 'offset', 'target'])

s_df = pd.read_csv(s_df_path)

# according to method, offset, target, print the rows that are in s_df but not in t_df

# Create a tuple key to compare rows
t_keys = set(zip(t_df['method'], t_df['offset'], t_df['target']))
s_keys = set(zip(s_df['method'], s_df['offset'], s_df['target']))

# Rows in static but not in trace
static_only_keys = s_keys - t_keys

print("in static only: " , len(static_only_keys))
for key in static_only_keys:
    print(key)

print("in trace only:" , len(t_keys - s_keys))
for key in t_keys - s_keys:
    print(key)