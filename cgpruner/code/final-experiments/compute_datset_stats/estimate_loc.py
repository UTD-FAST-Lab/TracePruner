import os

TRAINING_LIST = "../../../dataset-high-precision-callgraphs/train_programs.txt"
TEST_LIST = "../../../dataset-high-precision-callgraphs/test_programs.txt"
BENCHMARKS_FOLDER = "../../../dataset-high-precision-callgraphs/full_programs_set"


def get_directory_size(directory):
    """Returns the `directory` size in bytes."""
    total = 0
    try:
        # print("[+] Getting the size of", directory)
        for entry in os.scandir(directory):
            if entry.is_file():
                # if it's a file, use stat() function
                total += entry.stat().st_size
            elif entry.is_dir():
                # if it's a directory, recursively call this function
                total += get_directory_size(entry.path)
    except NotADirectoryError:
        # if `directory` isn't a directory, get the file size then
        return os.path.getsize(directory)
    except PermissionError:
        # if for whatever reason we can't open the folder, return 0
        return 0
    return total


'''
Compute the list of programs in the datset
'''
dataset = set()
with open(TRAINING_LIST) as f:
    for line in f:
        dataset.add(line.rstrip())

with open(TEST_LIST) as f:
    for line in f:
        dataset.add(line.rstrip())


for benchmark in os.listdir(BENCHMARKS_FOLDER):
    # only run for the dataset benchmarks
    if benchmark not in dataset:
        continue
    
    '''Compute LOC'''
    sources_file = BENCHMARKS_FOLDER + "/" + benchmark + "/info/sources"
    loc = 0
    with open(sources_file) as f1:
        for line in f1:
            filename = BENCHMARKS_FOLDER + "/" + benchmark + "/" + line.rstrip()
            with open(filename) as f2:
                lines2 = [line2.strip() for line2 in f2]
                loc += len(lines2)

    '''Compute sizes of class-files'''
    classes_folder = BENCHMARKS_FOLDER + "/" + benchmark + "/classes"
    classes_size = get_directory_size(classes_folder)

    lib_folder = BENCHMARKS_FOLDER + "/" + benchmark + "/lib"
    lib_size = get_directory_size(lib_folder)

    print(f'{benchmark},{loc},{classes_size},{lib_size}')
