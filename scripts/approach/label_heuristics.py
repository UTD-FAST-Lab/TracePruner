from collections import defaultdict, Counter

class LabelHeuristic:
    def label_clusters(self, labels, cluster_ids):
        raise NotImplementedError

class MajorityLabelHeuristic(LabelHeuristic):    #TODO: fix this for unknown labels.
    def label_clusters(self, labels, cluster_ids):
        cluster_to_labels = defaultdict(list)
        for lbl, cid in zip(labels, cluster_ids):
            if cid != -1:
                cluster_to_labels[cid].append(lbl)
        return {cid: Counter(vals).most_common(1)[0][0] for cid, vals in cluster_to_labels.items()}

class AnyTrueLabelHeuristic(LabelHeuristic):
    def label_clusters(self, labels, cluster_ids):
        cluster_labels = {}
        for lbl, cid in zip(labels, cluster_ids):
            if cid == -1:
                continue
            if cid not in cluster_labels:
                cluster_labels[cid] = lbl
            else:
                cluster_labels[cid] = cluster_labels[cid] or lbl
        return cluster_labels
