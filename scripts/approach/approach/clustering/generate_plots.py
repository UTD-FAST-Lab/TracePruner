from approach.data_representation.instance_loader import load_instances

from approach.utils import plot_points


instances = load_instances("njr")


plot_points(instances, "approach/results/plots/trace.png", feature_type='trace', plot_only_known=False, method='tsne')
plot_points(instances, "approach/results/plots/trace_known.png", feature_type='trace', plot_only_known=True, method='tsne')
plot_points(instances, "approach/results/plots/semantic.png", feature_type='semantic', plot_only_known=False, method='tsne')
plot_points(instances, "approach/results/plots/semantic_known.png", feature_type='semantic', plot_only_known=True, method='tsne')


# plot_points(instances, "results/plots/tsne/trace_sum_tfidf1.png", feature_type='trace', plot_only_known=False, method='tsne')

