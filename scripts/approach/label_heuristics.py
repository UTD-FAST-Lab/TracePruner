from collections import defaultdict, Counter

class LabelHeuristic:
    def label_clusters(self, labels, cluster_ids):
        raise NotImplementedError

class MajorityLabelHeuristic(LabelHeuristic):    
    def label_clusters(self, train, cluster_ids):
        cluster_to_labels = defaultdict(list)
        for inst, cid in zip(train, cluster_ids):
            if cid != -1 and inst.is_known():
                cluster_to_labels[cid].append(int(inst.get_label()))
        a =  {cid: Counter(vals).most_common(1)[0][0] for cid, vals in cluster_to_labels.items()}
        return a
    

class RelativeMajorityHeuristic(LabelHeuristic):
    def label_clusters(self, train, cluster_ids):

        # Compute total counts from labeled, known instances
        true_total = sum(1 for inst in train if inst.is_known() and inst.get_label() is True)
        false_total = sum(1 for inst in train if inst.is_known() and inst.get_label() is False)

        cluster_to_labels = defaultdict(list)
        for inst, cid in zip(train, cluster_ids):
            if cid != -1 and inst.is_known():
                cluster_to_labels[cid].append(int(inst.get_label()))

        cluster_to_label = {}
        for cid, labels in cluster_to_labels.items():
            label_counts = Counter(labels)
            true_prop = label_counts[1] / true_total if true_total > 0 else 0
            false_prop = label_counts[0] / false_total if false_total > 0 else 0
            cluster_to_label[cid] = 1 if true_prop >= false_prop else 0

        return cluster_to_label
    

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
