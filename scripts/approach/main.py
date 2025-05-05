import sys
sys.path.extend([".", ".."])


from approach.clustering_runner import ClusteringRunner
from approach.flat_clustering_runner import FlatClusteringRunner
from approach.rf_baseline import RandomForestBaseline
from approach.nn_baseline import NeuralNetBaseline
from approach.models.hdbscan_model import HDBSCANClusterer
from approach.models.mpckmeans_model import MPCKMeansClusterer
from approach.fallback import KNNFallback, CDistFallback, CDistFallback2
from approach.label_heuristics import MajorityLabelHeuristic, AnyTrueLabelHeuristic, RelativeMajorityHeuristic
from approach.instance_loader import load_instances

from approach.utils import plot_points

instances = load_instances("njr")

# plot_points(instances, "results/plots/struct.png", feature_type='static', plot_only_known=False, method='tsne')
# plot_points(instances, "results/plots/tsne/trace_sum_tfidf1.png", feature_type='trace', plot_only_known=False, method='tsne')


import os, pickle

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

# flat clustering
runner = FlatClusteringRunner(
    instances=instances,
    clusterer=HDBSCANClusterer(min_cluster_size=5),
    output_dir="results/hdbscan/flat/semantic",
    use_trace=False,
    use_semantic=True,
    use_static=False,
)


# cgpruner
# runner = RandomForestBaseline(instances=instances, threshold=0.45, train_with_unknown=True, make_balance=False , raw_baseline=True, output_dir="results/rf_baseline")

# autopruner
# runner = NeuralNetBaseline(
#     instances=instances,
#     raw_baseline=False,
#     train_with_unknown=False,
#     make_balance=False,
#     output_dir="results/nn_baseline",
#     use_trace=False,
#     use_semantic=False, 
#     use_static=True,
# )



runner.run()
