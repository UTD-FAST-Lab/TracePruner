import os
import time

CALLGRAPHS_FOLDER = "../final-experiments/overhead_measurement/wala_callgraphs"
RESULTS_FOLDER = "cgpruner_timings"
CLASSIFIER = "../../dataset-high-precision-callgraphs/learned_classifiers/wala.joblib"
SKIP_COMPLETED = True

#Loop through the benchmarks
for benchmark in os.listdir(CALLGRAPHS_FOLDER):
    print(benchmark)
    if (SKIP_COMPLETED):
        if os.path.exists(os.path.join(RESULTS_FOLDER,(benchmark+".txt"))):
            print("skipping completed benchmark.")
            continue
    
    #execute cgpruner on the callgraph
    start_time = time.time()
    command = (
        "python3 run_cg_pruner.py"
        + " --input "
        + CALLGRAPHS_FOLDER + "/" + benchmark
        + " --output temp/" + benchmark 
        + " --classifier "
        + CLASSIFIER
        + " --cutoff 0.45"
    )
    os.system(command)
    end_time = time.time()
    duration = end_time - start_time
    with open(RESULTS_FOLDER + "/" + benchmark, "w") as fp:
        fp.write(str(duration))
