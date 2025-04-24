import sys
sys.path.extend([".", ".."])


from approach.clustering_runner import ClusteringRunner
from approach.rf_baseline import RandomForestBaseline
from approach.nn_baseline import NeuralNetBaseline
from approach.models.hdbscan_model import HDBSCANClusterer
from approach.models.mpckmeans_model import MPCKMeansClusterer
from approach.fallback import KNNFallback, CDistFallback, CDistFallback2
from approach.label_heuristics import MajorityLabelHeuristic, AnyTrueLabelHeuristic, RelativeMajorityHeuristic
from approach.instance_loader import load_instances

# instances = load_instances("njr")

import os, pickle

if os.path.exists("cached_instances.pkl"):
    with open("cached_instances.pkl", "rb") as f:
        instances = pickle.load(f)
else:
    instances = load_instances("njr")
    with open("cached_instances.pkl", "wb") as f:
        pickle.dump(instances, f)

clusterers = (
    HDBSCANClusterer(min_cluster_size=5),
    MPCKMeansClusterer(n_clusters=2)
)

# runner = ClusteringRunner(
#     instances=instances,
#     clusterer=clusterers[1],
#     fallback=CDistFallback(),
#     labeler=RelativeMajorityHeuristic(),
#     output_dir="results/mpckmeans/relative_majority",
#     use_trace=False,
#     use_semantic=False,
#     use_static=False,
#     use_fallback=False,
#     train_with_unknown=True
# )

# cgpruner
# runner = RandomForestBaseline(instances=instances, threshold=0.45, train_with_unknown=True, make_balance=False , raw_baseline=True, output_dir="results/rf_baseline")

# autopruner
runner = NeuralNetBaseline(
    instances=instances,
    raw_baseline=False,
    train_with_unknown=False,
    make_balance=False,
    output_dir="results/nn_baseline",
    use_trace=False,
    use_semantic=False, 
    use_static=True,
)



runner.run()
