'''
This script just addds an extra column to a
csv file.
The script also adds the header line.
'''
import csv
import sys

with open(sys.argv[1]) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    print("method,offset,target,dir,tr")#print out header file
    for row in csv_reader:
        print(f'{row[0]},{row[1]},{row[2]},{row[3]},1')