import sys
sys.path.extend([".", ".."])


from approach.clustering_runner import ClusteringRunner
from approach.rf_baseline import RandomForestBaseline
from approach.models.hdbscan_model import HDBSCANClusterer
from approach.fallback import KNNFallback, CDistFallback
from approach.label_heuristics import MajorityLabelHeuristic, AnyTrueLabelHeuristic
from approach.instance_loader import load_instances

instances = load_instances("njr")

# runner = ClusteringRunner(
#     instances=instances,
#     clusterer=HDBSCANClusterer(min_cluster_size=5),
#     fallback=CDistFallback(),
#     labeler=AnyTrueLabelHeuristic(),
#     output_dir="results/any_normal",
#     use_trace=False,
#     use_fallback=True,
#     train_with_unknown=True
# )

runner = RandomForestBaseline(instances=instances, raw_baseline=True, output_dir="results/rf_baseline")

runner.run()
