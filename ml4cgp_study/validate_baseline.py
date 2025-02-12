import os
import csv
import pandas as pd

# Get the list of programs
programs = [program for program in os.listdir('/20TB/mohammad/cg_dataset/ml4cgp_study_data/xcorpus') 
            if program.endswith('.csv') and not program.startswith('xcorpus')]

# Open the output CSV file for writing
output_csv = '/20TB/mohammad/cg_dataset/output/xcorpus/concats/statistics/program_table_org.csv'

with open(output_csv, mode='w', newline='') as csvfile:
    fieldnames = ['program_name', 'precision', 'recall']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for program in programs:
        # Load the CSV file for each program
        csv_file = f'/20TB/mohammad/cg_dataset/ml4cgp_study_data/xcorpus/{program}'
        df = pd.read_csv(csv_file)

        # Drop rows that have the same 'method' and 'target' values
        df_unique = df.drop_duplicates(subset=['method', 'target'])

        # Count occurrences where 'wiretap' is 1
        wiretap_count = df_unique[df_unique['wiretap'] == 1].shape[0]
        wala_count = df_unique[df_unique['wala-cge-0cfa-noreflect-intf-trans'] == 1].shape[0]

        # Count occurrences where both 'wiretap' and 'wala' are 1
        wiretap_wala_count = df_unique[(df_unique['wiretap'] == 1) & (df_unique['wala-cge-0cfa-noreflect-intf-trans'] == 1)].shape[0]

        # Calculate recall and precision
        recall = wiretap_wala_count / wiretap_count if wiretap_count > 0 else 0
        precision = wiretap_wala_count / wala_count if wala_count > 0 else 0

        # Write the program's results to the output CSV file
        writer.writerow({
            'program_name': program,
            'precision': precision,
            'recall': recall
        })

        # Optionally print results
        # print(f"Program: {program}")
        # print(f"Number of times 'wiretap' is 1: {wiretap_count}")
        # print(f"Number of times both 'wiretap' and 'wala' are 1: {wiretap_wala_count}")
        # print(f"Recall: {recall}")
        # print(f"Precision: {precision}\n")
