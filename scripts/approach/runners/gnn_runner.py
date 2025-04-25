import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from approach.models.gnn_model import GCNGraphClassifier
from approach.utils import evaluate_fold, split_folds, write_metrics_to_csv
import pandas as pd
from tqdm import tqdm
import os

class GNNRunner:
    def __init__(self, instances, output_dir, hidden_dim=64, epochs=10, batch_size=32, lr=1e-3):
        self.instances = instances
        self.output_dir = output_dir
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.labeled = [i for i in instances if i.is_known()]
        self.unknown = [i for i in instances if not i.is_known()]

    def run(self):
        folds = split_folds(self.labeled, self.unknown, train_with_unknown=False)
        all_metrics = []
        all_embeddings = []

        for fold, (train, test) in enumerate(folds, 1):
            print(f"\n=== GNN Fold {fold} ===")

            train_data = [inst.get_trace_graph() for inst in train if inst.get_trace_graph() is not None]  #TODO: if the graph is None, do something.
            test_data = [inst.get_trace_graph() for inst in test if inst.get_trace_graph() is not None]

            for d, inst in zip(train_data, train):
                d.y = torch.tensor([1 if inst.get_label() else 0])
            for d, inst in zip(test_data, test):
                d.y = torch.tensor([1 if inst.get_label() else 0])

            model = GCNGraphClassifier(in_channels=train_data[0].x.shape[1], hidden_channels=self.hidden_dim).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = nn.CrossEntropyLoss()

            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

            self.train_loop(model, train_loader, optimizer, loss_fn)
            self.eval_loop(model, test_loader, test, all_embeddings)

            res = self.evaluate(test)
            metrics = res[0]
            metrics["unk_labeled_true"] = res[1]
            metrics["unk_labeled_false"] = res[2]
            metrics["unk_labeled_all"] = res[1] + res[2]

            gt_metrics = res[3]
            if gt_metrics:
                for k, v in gt_metrics.items():
                    metrics[f"gt_{k}"] = v

            metrics["fold"] = fold
            all_metrics.append(metrics)
            unk_labeled_true += res[1]
            unk_labeled_false += res[2]

            print(f"NN Fold {fold} - F1: {metrics['f1']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
            print(f"TP: {metrics['TP']} | FP: {metrics['FP']} | TN: {metrics['TN']} | FN: {metrics['FN']}")

        # Overall
       
        y_all_true = [1 if inst.get_label() else 0 for inst in self.labeled]
        y_all_pred = [1 if inst.get_predicted_label() else 0 for inst in self.labeled]

        overall = evaluate_fold(y_all_true, y_all_pred)
        overall["unk_labeled_true"] = unk_labeled_true
        overall["unk_labeled_false"] = unk_labeled_false
        overall["unk_labeled_all"] = unk_labeled_false + unk_labeled_true

        # Add evaluation on manually labeled unknowns
        gt_instances = [i for i in self.instances if i.ground_truth is not None and not i.is_known()]
        gt_y_true = [int(i.ground_truth) for i in gt_instances]
        gt_y_pred = [int(i.get_predicted_label()) for i in gt_instances]
        gt_metrics = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}

        for k, v in gt_metrics.items():
            overall[f"gt_{k}"] = v

        overall["fold"] = "overall"
        all_metrics.append(overall)

        print("\n=== NN Overall ===")
        print(f"Precision: {overall['precision']:.3f}, Recall: {overall['recall']:.3f}, F1: {overall['f1']:.3f}")
        print(f"TP: {overall['TP']} | FP: {overall['FP']} | TN: {overall['TN']} | FN: {overall['FN']}")

        write_metrics_to_csv(all_metrics, os.path.join(self.output_dir, "gnn_fold_metrics.csv"))
        self.save_embeddings(all_embeddings)


    def train_loop(self, model, loader, optimizer, loss_fn):
        model.train()
        for epoch in range(self.epochs):
            loop = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
            for batch in loop:
                batch = batch.to(self.device)
                out, _ = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                loss = loss_fn(out, batch.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    def eval_loop(self, model, loader, instances, embedding_accumulator):
        model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out, emb = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                probs = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())
                
                # Unbatch embeddings
                split_embs = torch.split(emb, 1)
                for i, (inst, e) in enumerate(zip(instances, split_embs)):
                    inst.set_predicted_label(int(preds[i]))
                    inst.set_confidence(float(probs[i]))
                    embedding_accumulator.append({
                        'program': inst.program,
                        'method': inst.src,
                        'offset': inst.offset,
                        'target': inst.target,
                        'embedding': e.squeeze(0).cpu().numpy().tolist()
                    })

    def save_embeddings(self, emb_list):
        df = pd.DataFrame(emb_list)
        df.to_csv(os.path.join(self.output_dir, "gnn_embeddings.csv"), index=False)



    def evaluate(self, test):
        y_true, y_pred, gt_y_true, gt_y_pred = [], [], [], []
        true, false = 0, 0

        for inst in test:
            pred = int(inst.get_predicted_label())
            if inst.is_known():
                y_true.append(int(inst.get_label()))
                y_pred.append(pred)
            else:
                true += pred
                false += (1 - pred)
                if inst.ground_truth is not None:
                    gt_y_true.append(int(inst.ground_truth))
                    gt_y_pred.append(pred)

        eval_main = evaluate_fold(y_true, y_pred)
        eval_gt = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}
        return eval_main, true, false, eval_gt
