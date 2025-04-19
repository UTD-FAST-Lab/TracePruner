import sys
sys.path.extend([".", ".."])


from approach.clustering_runner import ClusteringRunner
from approach.rf_baseline import RandomForestBaseline
from approach.models.hdbscan_model import HDBSCANClusterer
from approach.models.mpckmeans_model import MPCKMeansClusterer
from approach.fallback import KNNFallback, CDistFallback, CDistFallback2
from approach.label_heuristics import MajorityLabelHeuristic, AnyTrueLabelHeuristic, RelativeMajorityHeuristic
from approach.instance_loader import load_instances

instances = load_instances("njr")

clusterers = (
    HDBSCANClusterer(min_cluster_size=5),
    MPCKMeansClusterer(n_clusters=2)
)

runner = ClusteringRunner(
    instances=instances,
    clusterer=clusterers[0],
    fallback=CDistFallback(),
    labeler=MajorityLabelHeuristic(),
    output_dir="results/majority/hdbscan",
    use_trace=False,
    use_fallback=True,
    train_with_unknown=True
)

# runner = RandomForestBaseline(instances=instances, threshold=0.45, train_with_unknown=False, make_balance=True , raw_baseline=False, output_dir="results/rf_baseline")

runner.run()
