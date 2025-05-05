# import os
# import json
# import csv

# base_dir = '/20TB/mohammad/data/variables'
# output_dir = '/20TB/mohammad/data/var_repr'
# os.makedirs(output_dir, exist_ok=True)

# for program in os.listdir(base_dir):
#     var_info_dir = os.path.join(base_dir, program)
#     var_path = os.path.join(var_info_dir, 'var.txt')

#     if not os.path.exists(var_path):
#         continue

#     with open(var_path, 'r') as f:
#         lines = [line.strip() for line in f if line.strip()]

#     # Group logs by instruction
#     data_by_instruction = {}

#     for line in lines:
#         try:
#             tag, json_str = line.split(':', 1)
#             data = json.loads(json_str.strip())
#             instr = data.get('id')
#             if not instr:
#                 continue

#             if instr not in data_by_instruction:
#                 data_by_instruction[instr] = {}

#             if 'POINTER_INFO_visitInvoke' in tag:
#                 data_by_instruction[instr]['pointer_visit'] = data
#             elif 'POINTER_INFO_addEdge' in tag:
#                 data_by_instruction[instr]['pointer_add'] = data
#             elif 'CALL_GRAPH_INFO_visitInvoke' in tag:
#                 data_by_instruction[instr]['callgraph_visit'] = data
#             elif 'CALL_GRAPH_INFO_addEdge' in tag:
#                 data_by_instruction[instr]['callgraph_add'] = data

#         except Exception as e:
#             print(f"[{program}] Skipping bad line: {line[:80]} -- {e}")

#     # Extract and save matched entries
#     rows = []
#     for instr, parts in data_by_instruction.items():
#         if not all(k in parts for k in ['pointer_visit', 'pointer_add', 'callgraph_visit', 'callgraph_add']):
#             continue

#         # Skip if src_node is "null"
#         if parts['callgraph_add'].get('src_node') == "null":
#             continue

#         row = {

#              # Add direct src/target/offset fields (redundant, but easier for downstream)
#             'src_node': parts['callgraph_add'].get('src_node'),
#             'offset': parts['callgraph_add'].get('offset'),
#             'target_node': parts['callgraph_add'].get('target_node'),


#             # POINTER_INFO visit
#             'receiver_visit': parts['pointer_visit'].get('receiver'),
#             'parameters_visit': json.dumps(parts['pointer_visit'].get('parameters')),

#             # POINTER_INFO addEdge
#             'receiver_add': parts['pointer_add'].get('receiver'),
#             'parameters_add': json.dumps(parts['pointer_add'].get('parameters')),

#             # CALL_GRAPH_INFO visit
#             'call_visit_method': parts['callgraph_visit'].get('current_node', {}).get('method'),
#             'call_visit_out': parts['callgraph_visit'].get('current_node', {}).get('out_edges'),
#             'call_visit_in': parts['callgraph_visit'].get('current_node', {}).get('in_edges'),

#             # CALL_GRAPH_INFO addEdge
#             'call_add_method': parts['callgraph_add'].get('current_node', {}).get('method'),
#             'call_add_out': parts['callgraph_add'].get('current_node', {}).get('out_edges'),
#             'call_add_in': parts['callgraph_add'].get('current_node', {}).get('in_edges'),

           
#         }
#         rows.append(row)

#     if rows:
#         out_path = os.path.join(output_dir, f'{program}_full_info.csv')
#         with open(out_path, 'w', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=rows[0].keys())
#             writer.writeheader()
#             writer.writerows(rows)



# import os
# import json
# import csv

# base_dir = '/20TB/mohammad/data/variables'
# output_dir = '/20TB/mohammad/data/var_repr'
# os.makedirs(output_dir, exist_ok=True)

# for program in os.listdir(base_dir):
#     var_info_dir = os.path.join(base_dir, program)
#     var_path = os.path.join(var_info_dir, 'var.txt')

#     if not os.path.exists(var_path):
#         continue

#     with open(var_path, 'r') as f:
#         lines = [line.strip() for line in f if line.strip()]

#     # Separate visit and addEdge messages
#     visit_data = {}
#     add_edges_by_id = {}

#     for line in lines:
#         try:
#             tag, json_str = line.split(':', 1)
#             data = json.loads(json_str.strip())
#             call_id = data.get('id')
#             if not call_id:
#                 continue

#             if 'visitInvoke' in tag:
#                 # Overwrite previous visitInvoke info (latest one wins)
#                 if call_id not in visit_data:
#                     visit_data[call_id] = {}
#                 if 'POINTER_INFO_visitInvoke' in tag:
#                     visit_data[call_id]['pointer'] = data
#                 elif 'CALL_GRAPH_INFO_visitInvoke' in tag:
#                     visit_data[call_id]['callgraph'] = data

#             elif 'addEdge' in tag:
#                 if call_id not in add_edges_by_id:
#                     add_edges_by_id[call_id] = []
#                 add_edges_by_id[call_id].append((tag, data))

#         except Exception as e:
#             print(f"[{program}] Skipping bad line: {line[:80]} -- {e}")

#     # Now combine visit info with each addEdge
#     rows = []
#     for call_id, add_edges in add_edges_by_id.items():
#         visit = visit_data.get(call_id)
#         if not visit or 'pointer' not in visit or 'callgraph' not in visit:
#             continue  # skip incomplete

#         for tag, add_data in add_edges:
#             if add_data.get('src_node') == "null":
#                 continue

#             row = {
#                 # Shared call site info
#                 'src_node': add_data.get('src_node'),
#                 'offset': add_data.get('offset'),
#                 'target_node': add_data.get('target_node'),

#                 # visitInvoke pointer info
#                 'receiver_visit': visit['pointer'].get('receiver'),
#                 'parameters_visit': json.dumps(visit['pointer'].get('parameters')),

#                 # addEdge pointer info (if present)
#                 'receiver_add': add_data.get('receiver'),
#                 'parameters_add': json.dumps(add_data.get('parameters')) if 'POINTER_INFO_addEdge' in tag else "",

#                 # visitInvoke call graph info
#                 'call_visit_method': visit['callgraph'].get('current_node', {}).get('method'),
#                 'call_visit_out': visit['callgraph'].get('current_node', {}).get('out_edges'),
#                 'call_visit_in': visit['callgraph'].get('current_node', {}).get('in_edges'),

#                 # addEdge call graph info
#                 'call_add_method': add_data.get('current_node', {}).get('method') if 'CALL_GRAPH_INFO_addEdge' in tag else "",
#                 'call_add_out': add_data.get('current_node', {}).get('out_edges') if 'CALL_GRAPH_INFO_addEdge' in tag else "",
#                 'call_add_in': add_data.get('current_node', {}).get('in_edges') if 'CALL_GRAPH_INFO_addEdge' in tag else "",
#             }

#             rows.append(row)

#     if rows:
#         out_path = os.path.join(output_dir, f'{program}_full_info.csv')
#         with open(out_path, 'w', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=rows[0].keys())
#             writer.writeheader()
#             writer.writerows(rows)


import os
import json
import csv

base_dir = '/20TB/mohammad/data/variables'
output_dir = '/20TB/mohammad/data/var_repr'
os.makedirs(output_dir, exist_ok=True)

for program in os.listdir(base_dir):
    var_info_dir = os.path.join(base_dir, program)
    var_path = os.path.join(var_info_dir, 'var.txt')

    if not os.path.exists(var_path):
        continue

    with open(var_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    visit_data = {}
    pointer_add_map = {}
    callgraph_add_map = {}

    for line in lines:
        try:
            tag, json_str = line.split(':', 1)
            data = json.loads(json_str.strip())
            call_id = data.get('id')
            if not call_id:
                continue

            if 'visitInvoke' in tag:
                if call_id not in visit_data:
                    visit_data[call_id] = {}
                if 'POINTER_INFO_visitInvoke' in tag:
                    visit_data[call_id]['pointer'] = data
                elif 'CALL_GRAPH_INFO_visitInvoke' in tag:
                    visit_data[call_id]['callgraph'] = data

            elif 'POINTER_INFO_addEdge' in tag:
                key = (call_id, data.get('target_node'), data.get('offset'))
                pointer_add_map[key] = data

            elif 'CALL_GRAPH_INFO_addEdge' in tag:
                key = (call_id, data.get('target_node'), data.get('offset'))
                callgraph_add_map[key] = data

        except Exception as e:
            print(f"[{program}] Skipping bad line: {line[:80]} -- {e}")

    # Merge matching addEdge entries and enrich with visit info
    rows = []
    for key in set(pointer_add_map.keys()).intersection(callgraph_add_map.keys()):
        call_id, target, offset = key
        pointer_add = pointer_add_map[key]
        callgraph_add = callgraph_add_map[key]

        if callgraph_add.get('src_node') == "null" or callgraph_add.get('target_node') == "null":
            continue

        visit = visit_data.get(call_id)
        if not visit or 'pointer' not in visit or 'callgraph' not in visit:
            continue

        row = {
            # Basic edge info
            'src_node': callgraph_add.get('src_node'),
            'offset': offset,
            'target_node': target,

            # visitInvoke POINTER info
            'receiver_visit': visit['pointer'].get('receiver'),
            'parameters_visit': json.dumps(visit['pointer'].get('parameters')),

            # addEdge POINTER info
            'receiver_add': pointer_add.get('receiver'),
            'parameters_add': json.dumps(pointer_add.get('parameters')),

            # visitInvoke CALLGRAPH info
            'call_visit_method': visit['callgraph'].get('current_node', {}).get('method'),
            'call_visit_out': visit['callgraph'].get('current_node', {}).get('out_edges'),
            'call_visit_in': visit['callgraph'].get('current_node', {}).get('in_edges'),

            # addEdge CALLGRAPH info
            'call_add_method': callgraph_add.get('current_node', {}).get('method'),
            'call_add_out': callgraph_add.get('current_node', {}).get('out_edges'),
            'call_add_in': callgraph_add.get('current_node', {}).get('in_edges'),
        }

        rows.append(row)

    if rows:
        out_path = os.path.join(output_dir, f'{program}_full_info.csv')
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
