import sys
# sys.path.extend([".", ".."])

import os, pickle

from approach.clustering.clustering_runner import ClusteringRunner
from approach.clustering.flat_clustering_runner import FlatClusteringRunner
from approach.models.hdbscan_model import HDBSCANClusterer
from approach.models.mpckmeans_model import MPCKMeansClusterer
from approach.clustering.fallback import KNNFallback, CDistFallback, CDistFallback2
from approach.clustering.label_heuristics import MajorityLabelHeuristic, AnyTrueLabelHeuristic, RelativeMajorityHeuristic
from approach.data_representation.instance_loader import load_instances

from approach.utils import plot_points



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
        
   

if __name__ == "__main__":
    main()
