import sys
# sys.path.extend([".", ".."])

import os, pickle
from collections import defaultdict

from approach.clustering.clustering_runner import ClusteringRunner
from approach.clustering.flat_clustering_runner import FlatClusteringRunner
from approach.models.hdbscan_model import HDBSCANClusterer
from approach.models.mpckmeans_model import MPCKMeansClusterer
from approach.clustering.fallback import KNNFallback, CDistFallback, CDistFallback2
from approach.clustering.label_heuristics import MajorityLabelHeuristic, AnyTrueLabelHeuristic, RelativeMajorityHeuristic
# from approach.data_representation.instance_loader import load_instances  #TODO: fix this mess
from approach.data_representation.instance_loader_xcorp import load_instances

from approach.utils import plot_points

import argparse
parser = argparse.ArgumentParser(description="Run baseline models")
parser.add_argument("--exp", type=str, default="1", help="Type of experiment to run (1,2,3)")

# plot_points(instances, "results/plots/struct.png", feature_type='static', plot_only_known=False, method='tsne')
# plot_points(instances, "results/plots/tsne/trace_sum_tfidf1.png", feature_type='trace', plot_only_known=False, method='tsne')



# if os.path.exists("cached_instances_trace.pkl"):
#     with open("cached_instances_trace.pkl", "rb") as f:
#         instances = pickle.load(f)
# else:
#     instances = load_instances("njr")
#     with open("cached_instances_trace.pkl", "wb") as f:
#         pickle.dump(instances, f)

# clusterers = (
#     HDBSCANClusterer(min_cluster_size=5),
#     MPCKMeansClusterer(n_clusters=2)
# )

# runner = ClusteringRunner(
#     instances=instances,
#     clusterer=clusterers[0],
#     fallback=CDistFallback2(),
#     labeler=AnyTrueLabelHeuristic(),
#     output_dir="results/hdbscan/trace/any_normal",
#     use_trace=True,
#     use_semantic=False,
#     use_static=False,
#     use_fallback=True,
#     train_with_unknown=True
# )




def main():

    # Load instances
    instances = load_instances("njr")
    all_runners = []

    for i in range(2):

        # struct
        runner = FlatClusteringRunner(
            instances=instances,
            clusterer=HDBSCANClusterer(min_cluster_size=5),
            only_true=bool(i),
            output_dir="approach/results/clustering",
            use_trace=False,
            use_semantic=False,
            use_static=True,
            run_from_main=True,
        )
        all_runners.append(runner)

        # semantic
        runner = FlatClusteringRunner(
            instances=instances,
            clusterer=HDBSCANClusterer(min_cluster_size=5),
            only_true=bool(i),
            output_dir="approach/results/clustering",
            use_trace=False,
            use_semantic=True,
            use_static=False,
            run_from_main=True,
        )
        all_runners.append(runner)

        # trace
        runner = FlatClusteringRunner(
            instances=instances,
            clusterer=HDBSCANClusterer(min_cluster_size=5),
            only_true=bool(i),
            output_dir="approach/results/clustering",
            use_trace=True,
            use_semantic=False,
            use_static=False,
            run_from_main=True,
        )
        all_runners.append(runner)


    for runner in all_runners:
        runner.run()
        
   

def main_xcorp(args):

    clusterer = HDBSCANClusterer(min_cluster_size=5, metric='hamming', alpha=0.5)

    if args.exp == "1":
        params = [
            ("doop",("v1", "39"),False),
            ("doop",("v3", "5"),False),
            ("doop",("v2", "0"),False),
            ("wala",("v1", "19"),False),
            ("wala",("v3", "0"),False),
            ("wala",("v1", "23"),False),
            ("opal",("v1", "0"),False),
        ]

        for param in params:
            instances = load_instances(tool=param[0], config_info=param[1], just_three=True)
             # seperate instances by thrir program
            program_instances = defaultdict(list)
            for inst in instances:
                program_instances[inst.program].append(inst)
            
            # For each program, run the clustering
            for program, insts in program_instances.items():
                print(f"Running clustering for program: {program} with {len(insts)} instances")
                if len(insts) < 5:
                    print(f"Skipping program {program} due to insufficient instances ({len(insts)})")
                    continue
                output_dir = f"approach/results/clustering/programwise/{param[0]}/{param[1][0]}_{param[1][1]}/{program}"
                os.makedirs(output_dir, exist_ok=True)

                # Create a runner for each program
                runner = FlatClusteringRunner(
                    instances=insts,
                    clusterer=clusterer,
                    output_dir=output_dir,
                    run_from_main=True,
                    only_true=False,
                    use_trace=False,
                    use_var=False,
                    use_semantic=False,
                    use_static=True,
                )
                
                # Run the clustering
                runner.run()


    elif args.exp == "2":
        for tool in ['wala', 'doop']:
            instances = load_instances(tool=tool, config_info=None, just_three=True)
             # seperate instances by thrir program
            program_instances = defaultdict(list)
            for inst in instances:
                program_instances[inst.program].append(inst)
            
            # For each program, run the clustering
            for program, insts in program_instances.items():
                print(f"Running clustering for program: {program} with {len(insts)} instances")
                if len(insts) < 5:
                    print(f"Skipping program {program} due to insufficient instances ({len(insts)})")
                    continue
                output_dir = f"approach/results/clustering/programwise/{tool}/{program}"
                os.makedirs(output_dir, exist_ok=True)

                # Create a runner for each program
                runner = FlatClusteringRunner(
                    instances=insts,
                    clusterer=clusterer,
                    output_dir=output_dir,
                    run_from_main=True,
                    only_true=False,
                    use_trace=False,
                    use_var=False,
                    use_semantic=False,
                    use_static=True,
                )
                
                # Run the clustering
                runner.run()

    elif args.exp == "3":
        instances = load_instances(tool=None, config_info=None, just_three=True)
        # seperate instances by thrir program
        program_instances = defaultdict(list)
        for inst in instances:
            program_instances[inst.program].append(inst)
        
        # For each program, run the clustering
        for program, insts in program_instances.items():
            print(f"Running clustering for program: {program} with {len(insts)} instances")
            if len(insts) < 5:
                print(f"Skipping program {program} due to insufficient instances ({len(insts)})")
                continue
            output_dir = f"approach/results/clustering/programwise/{program}"
            os.makedirs(output_dir, exist_ok=True)

            # Create a runner for each program
            runner = FlatClusteringRunner(
                instances=insts,
                clusterer=clusterer,
                output_dir=output_dir,
                run_from_main=True,
                only_true=False,
                use_trace=False,
                use_var=False,
                use_semantic=False,
                use_static=True,
            )
            
            # Run the clustering
            runner.run()

    else:
        raise ValueError("Invalid experiment type specified. Use 1, 2, or 3.")


    
   





if __name__ == "__main__":
    # main() #njr
    main_xcorp(parser.parse_args())
