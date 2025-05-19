import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from approach.utils import evaluate_fold, write_results_to_csv, write_metrics_to_csv, split_folds, balance_training_set
from approach.models.nn_models import NNClassifier_Combine, NNClassifier_Semantic, NNClassifier_Structure
import torch.nn.functional as F
from tqdm import tqdm

class NeuralNetBaseline:
    def __init__(self, instances, output_dir, train_with_unknown=True, make_balance=False, threshold=0.5, raw_baseline=False,
                 use_trace=False, use_semantic=False, use_static=False, hidden_size=32, batch_size=100, lr=5e-6, epochs=5):
        self.instances = instances
        self.raw_baseline = raw_baseline
        self.output_dir = output_dir
        self.use_trace = use_trace
        self.use_semantic = use_semantic
        self.use_static = use_static
        self.make_balance = make_balance
        self.train_with_unknown = train_with_unknown
        self.threshold = threshold
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.labeled = [i for i in instances if i.is_known()]
        self.unknown = [i for i in instances if not i.is_known()]

        input_mode = (use_semantic, use_static)
        if input_mode == (True, False):
            self.model = NNClassifier_Semantic(hidden_size)
        elif input_mode == (False, True):
            self.model = NNClassifier_Structure(hidden_size)
        else:
            self.model = NNClassifier_Combine(hidden_size)
        self.model.to(self.device)

    def get_features(self, insts):
        code_vecs = []
        struct_vecs = []

        for inst in insts:
            if self.use_semantic and not self.use_static:
                code = inst.get_semantic_features()
                struct = [0.0] * 11
            elif self.use_static and not self.use_semantic:
                code = [0.0] * 768
                struct = inst.get_static_featuers()
            else:
                code = inst.get_semantic_features()
                struct = inst.get_static_featuers()

            # Validate vector lengths
            if code is None or len(code) != 768:
                code = [0.0] * 768
            if struct is None or len(struct) != 11:
                print(f"warning: instance structured data is not correct")
                struct = [0.0] * 11

            code_vecs.append(code)
            struct_vecs.append(struct)

        return np.array(code_vecs, dtype=np.float32), np.array(struct_vecs, dtype=np.float32)

    def run(self):
        folds = split_folds(self.labeled, self.unknown, self.train_with_unknown)
        all_metrics = []
        unk_labeled_true = 0
        unk_labeled_false = 0

        for fold, (train, test) in enumerate(folds, 1):
            print(f"\n=== NN Fold {fold} ===")

            if self.make_balance:
                if self.make_balance[0] == "smote":
                    return
                else:
                    train = balance_training_set(train, self.make_balance[0], self.make_balance[1])

            code_train, struct_train = self.get_features(train)
            y_train = np.array([1 if i.get_label() else 0 for i in train])

            code_test, struct_test = self.get_features(test)
            y_test = np.array([1 if i.get_label() else 0 for i in test])

            # Create loaders
            train_loader = self.create_loader(code_train, struct_train, y_train, train, shuffle=True)
            test_loader = self.create_loader(code_test, struct_test, y_test, test, shuffle=False)

            # Reinit model and optimizer
            self.model.apply(self.init_weights)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            loss_fn = nn.CrossEntropyLoss()

            self.train_loop(train_loader, optimizer, loss_fn)
            self.eval_loop(test_loader, test)

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
        if not self.raw_baseline:
            y_all_true = [1 if inst.get_label() else 0 for inst in self.labeled]
            y_all_pred = [1 if inst.get_predicted_label() else 0 for inst in self.labeled]
        else:
            y_all_true = [1 if inst.get_label() else 0 for inst in self.instances]
            y_all_pred = [1 if inst.get_predicted_label() else 0 for inst in self.instances]

        overall = evaluate_fold(y_all_true, y_all_pred)
        overall["unk_labeled_true"] = unk_labeled_true
        overall["unk_labeled_false"] = unk_labeled_false
        overall["unk_labeled_all"] = unk_labeled_false + unk_labeled_true

        # Add evaluation on manually labeled unknowns
        # gt_instances = [i for i in self.instances if i.ground_truth is not None and not i.is_known()]
        # gt_y_true = [int(i.ground_truth) for i in gt_instances]
        # gt_y_pred = [int(i.get_predicted_label()) for i in gt_instances]
        # gt_metrics = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}

        # for k, v in gt_metrics.items():
        #     overall[f"gt_{k}"] = v

        # calculate the mean of all the gt metrics
        for metric in all_metrics:
            for k, v in metric.items():
                if k.startswith("gt_"):
                    if k not in overall:
                        overall[k] = 0
                    overall[k] += v
        for k in overall.keys():
            if k.startswith("gt_"):
                overall[k] =  round(overall[k]/len(all_metrics),3)

        overall["fold"] = "overall"
        all_metrics.append(overall)

        print("\n=== NN Overall ===")
        print(f"Precision: {overall['precision']:.3f}, Recall: {overall['recall']:.3f}, F1: {overall['f1']:.3f}")
        print(f"TP: {overall['TP']} | FP: {overall['FP']} | TN: {overall['TN']} | FN: {overall['FN']}")

        if not self.raw_baseline:
            if self.train_with_unknown:
                metrics_path = f"{self.output_dir}/nn_{self.threshold}_trained_on_unknown.csv"
            elif self.make_balance:
                metrics_path = f"{self.output_dir}/nn_{self.threshold}_trained_on_known_{self.make_balance[0]}_{self.make_balance[1]}.csv"
            else:
                metrics_path = f"{self.output_dir}/nn_{self.threshold}_trained_on_known.csv"
        else:
            metrics_path = f"{self.output_dir}/nn_raw_{self.threshold}.csv"

        # write_results_to_csv(all_eval, res_path)
        write_metrics_to_csv(all_metrics, metrics_path)

    def create_loader(self, code, struct, y, instances, shuffle=True):
        code_tensor = torch.tensor(code, dtype=torch.float32)
        struct_tensor = torch.tensor(struct, dtype=torch.float32)
        label_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(code_tensor, struct_tensor, label_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_loop(self, loader, optimizer, loss_fn):
        self.model.train()
        for epoch in range(self.epochs):
            loop = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
            for code, struct, label in loop:
                code, struct, label = code.to(self.device), struct.to(self.device), label.to(self.device)
                output = self.model(code, struct)
                loss = loss_fn(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def eval_loop(self, loader, test_instances):
        self.model.eval()
        all_preds = []
        all_confs = []
        with torch.no_grad():
            for code, struct, _ in loader:
                code, struct = code.to(self.device), struct.to(self.device)
                output = self.model(code, struct)
                probs = F.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
                preds = (probs >= self.threshold).astype(int)
                all_preds.extend(preds.tolist())
                all_confs.extend(probs.tolist())
        for inst, pred, conf in zip(test_instances, all_preds, all_confs):
            inst.set_predicted_label(pred)
            inst.set_confidence(conf)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def evaluate(self, test):
        y_true, y_pred, gt_y_true, gt_y_pred = [], [], [], []
        true, false = 0, 0

        for inst in test:
            pred = int(inst.get_predicted_label())
            if inst.is_known():
                y_true.append(int(inst.get_label()))
                y_pred.append(pred)
            else:
                if inst.ground_truth is not None:
                    gt_y_true.append(int(inst.ground_truth))
                    gt_y_pred.append(pred)
                else:   
                    true += pred
                    false += (1 - pred)

        eval_main = evaluate_fold(y_true, y_pred)
        eval_gt = evaluate_fold(gt_y_true, gt_y_pred) if gt_y_true else {}
        return eval_main, true, false, eval_gt

