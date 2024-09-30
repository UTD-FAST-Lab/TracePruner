import sys
import pathlib
import os

def file_len(fname):
    with open(fname) as f:
        content = f.read().split("\n") 
        return len(content)

WALA_FOLDER = pathlib.Path(sys.argv[1])
BALANCER_FOLDER = pathlib.Path(sys.argv[2])

wala_files = {}
balancer_files = {}

for wf in WALA_FOLDER.iterdir():
    wala_files[wf.name] = wf

for bf in BALANCER_FOLDER.iterdir():
    balancer_len = file_len(str(bf)) - 2
    wala_len = file_len(str(wala_files[bf.name])) - 2
    if (wala_len != -1 and balancer_len != -1):
        #-1 means the benchmarks didn't complete in the time limit.
        print(f'{bf.name},{wala_len},{balancer_len}')
