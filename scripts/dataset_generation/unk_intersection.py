import os
import pandas as pd

dataset_dir = "/20TB/mohammad/xcorpus-total-recall/dataset"
manual_dir = "/20TB/mohammad/xcorpus-total-recall/manual_labeling"

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


def overall():

    for program in programs:
        total_unknowns = []
        
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

                total_unknowns.append(unknown_df)
        

        if total_unknowns:
            # calculate the intersection of all unknowns
            intersection_df = total_unknowns[0]
            for df in total_unknowns[1:]:
                intersection_df = pd.merge(intersection_df, df, on=['method', 'offset', 'target'], how='inner')
            intersection_df = intersection_df.drop_duplicates(subset=['method', 'offset', 'target'])
            # Save the intersection to a CSV file
            intersection_path = os.path.join(manual_dir, 'overall', program, f'unknown_{program}.csv')
            os.makedirs(os.path.dirname(intersection_path), exist_ok=True)
            intersection_df.to_csv(intersection_path, index=False)


def toolwise():


    for program in programs:
        
        for tool in tools:
            total_unknowns = []
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

                total_unknowns.append(unknown_df)
        

            if total_unknowns:
                # calculate the intersection of all unknowns
                intersection_df = total_unknowns[0]
                for df in total_unknowns[1:]:
                    intersection_df = pd.merge(intersection_df, df, on=['method', 'offset', 'target'], how='inner')
                intersection_df = intersection_df.drop_duplicates(subset=['method', 'offset', 'target'])
                # Save the intersection to a CSV file
                intersection_path = os.path.join(manual_dir, tool, program, f'unknown_{program}.csv')
                os.makedirs(os.path.dirname(intersection_path), exist_ok=True)
                intersection_df.to_csv(intersection_path, index=False)



def main():
    
    overall()

    toolwise()


if __name__ == "__main__":
    main()