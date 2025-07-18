import os
import pandas as pd


compare_dir = '/20TB/mohammad/xcorpus-total-recall/compare'
output_dir = '/20TB/mohammad/xcorpus-total-recall/false_edges'


tools = [
    'wala',
    'doop'
]

programs = [
    'axion',
    'batik',
    'jasml',
    'xerces'
]


def main():

    for program in programs:
        false_dfs = []
        for tool in tools:
            program_dir = os.path.join(compare_dir,tool, 'final', program)
            for file in os.listdir(program_dir):
                false_df = pd.read_csv(os.path.join(program_dir, file)).drop_duplicates()
                false_dfs.append(false_df)

        # merge all the false edges of the program and drop duplicates and save to output directory
        if false_dfs:
            merged_df = pd.concat(false_dfs).drop_duplicates()
            output_file = os.path.join(output_dir, program, "false_edges.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            merged_df.to_csv(output_file, index=False)
            print(f"Saved false edges for {program} to {output_file}")


if __name__ == "__main__":
    main()