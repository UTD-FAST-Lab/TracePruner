
import os
import pandas as pd
dataset_dir = "/20TB/mohammad/xcorpus-total-recall/dataset"

programs = [
    'axion',
    'batik',
    'xerces',
    'jasml'
]

tools = [
    'wala',
    'doop',
    'opal'
]

from upsetplot import UpSet, from_memberships
import matplotlib.pyplot as plt

def plot_upset_for_selected_unknowns(total_unknowns_dict, program, selected_keys):
    sets = {}
    for tool, config in selected_keys:
        df = total_unknowns_dict.get((tool, config))
        if df is not None and not df.empty:
            label = f"{tool}_{config}"
            edge_ids = set(df[['method', 'offset', 'target']].astype(str).agg('|'.join, axis=1))
            sets[label] = edge_ids

    # Build memberships
    all_edges = set().union(*sets.values())
    memberships = []
    for edge in all_edges:
        present_in = [label for label, s in sets.items() if edge in s]
        memberships.append(present_in)

    # Build and plot
    data = from_memberships(memberships)
    UpSet(data, subset_size='count', show_counts=True).plot()
    plt.suptitle(f"Unknown Edge Intersections for {program}")
    plt.savefig(f"unknown_edges_intersections_{program}.png")



def calculate_total_unknowns(total_unknowns_dict, program):
    total_unknowns = list(total_unknowns_dict.values())

        # union of total unknowns
    if total_unknowns:
        union_df = pd.concat(total_unknowns).drop_duplicates(subset=['method', 'offset', 'target'])
        
        print(f"Total unknown edges for {program}: {len(union_df)}")
    
    # intersection of total unknowns
    if total_unknowns:
        # calculate the intersection of all unknowns
        intersection_df = total_unknowns[0]
        for df in total_unknowns[1:]:
            intersection_df = pd.merge(intersection_df, df, on=['method', 'offset', 'target'], how='inner')
        intersection_df = intersection_df.drop_duplicates(subset=['method', 'offset', 'target'])
        print(f"Total unknown edges intersection for {program}: {len(intersection_df)}")



def calculate_selected_unknowns(total_unknowns_dict, program):
    selected_keys = [
        # ('wala', 'v1_4'),  #rta_full
        ('wala', 'v1_19'), #0cfa,ONE_FLOW_TO_CASTS_NO_METHOD_INVOKE (excluding xerces)   
        ('wala', 'v3_0'),  #1cfa,ONE_FLOW_TO_CASTS_NO_METHOD_INVOKE (excluding xerces) 
        # ('wala', 'v2_18'),  #1obj,String_only (excluding xerces) 
        ('wala', 'v1_23'),  #0cfa,String_only (excluding xerces) 

        ('doop', 'v1_39'),  #0cfa_on (excluding jasml)
        ('doop', 'v3_5'),   #1_type, on
        # ('doop', 'v1_3'),  # 1obj,off (excluding xerces)
        ('doop', 'v2_0'),  # 1obj,off (excluding xerces)

        ('opal', 'v1_0'),   #cha  
        # ('opal', 'v1_8'),   #0-1cfa  
        # ('opal', 'v1_9'),    #11cfa (excluding axion and maybe jasml )
    ]

    # plot_upset_for_selected_unknowns(total_unknowns_dict, program, selected_keys)

    selected_unknowns = [v for k, v in total_unknowns_dict.items() if k in selected_keys]


    # union of total unknowns
    if selected_unknowns:
        union_df = pd.concat(selected_unknowns).drop_duplicates(subset=['method', 'offset', 'target'])
        
        print(f"Total unknown edges for {program}: {len(union_df)}")


    # intersection of total unknowns
    if selected_unknowns:
        # calculate the intersection of all unknowns
        intersection_df = selected_unknowns[0]
        for df in selected_unknowns[1:]:
            intersection_df = pd.merge(intersection_df, df, on=['method', 'offset', 'target'], how='inner')
        intersection_df = intersection_df.drop_duplicates(subset=['method', 'offset', 'target'])
        print(f"Total unknown edges intersection for {program}: {len(intersection_df)}")




def main():

    for program in programs:
        total_unknowns_dict = {}
        
        for tool in tools:
            tool_dir = os.path.join(dataset_dir, tool, 'without_jdk', program)
            if not os.path.exists(tool_dir):
                continue

            for config in os.listdir(tool_dir):
                config_dir = os.path.join(tool_dir, config)
                if not os.path.isdir(config_dir):
                    continue

                # Read the unknowns file
                unknown_file = os.path.join(config_dir, 'unknown_edges.csv')
                if not os.path.exists(unknown_file):
                    continue
                
                unknown_df = pd.read_csv(unknown_file).drop_duplicates(subset=['method', 'offset', 'target'])

                if unknown_df.empty:
                    continue

                # total_unknowns.append(unknown_df)
                total_unknowns_dict[(tool, config)] = unknown_df
        


        # calculate_total_unknowns(total_unknowns_dict, program)
        # print("**" * 20)
        calculate_selected_unknowns(total_unknowns_dict, program)



if __name__ == '__main__':
    main()