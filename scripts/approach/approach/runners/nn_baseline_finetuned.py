import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from approach.utils import evaluate_fold, write_results_to_csv, write_metrics_to_csv, write_instances_to_file ,split_folds, balance_training_set, split_folds_programs, split_fixed_set
from approach.models.fine_tuning.llm_models import CodeBERTClassifier, CodeT5Classifier
import torch.nn.functional as F
from tqdm import tqdm

MODELS_BATCH_SIZE = {"codebert": 32, "codet5": 12}
NO_EPOCHS = 2
LR = 0.00001

class NeuralNetBaselineFineTuned:
    def __init__(self, instances, output_dir, train_with_unknown=True, make_balance=False, threshold=0.5, raw_baseline=False,
                 use_trace=False, use_semantic=False, use_static=False, model_name='codebert' , hidden_size=32, batch_size=100, lr=5e-6, epochs=5, just_three=False, random_split=False):
        self.instances = instances
        self.raw_baseline = raw_baseline
        self.output_dir = output_dir
        self.use_trace = use_trace
        self.use_semantic = use_semantic
        self.use_static = use_static
        self.make_balance = make_balance
        self.train_with_unknown = train_with_unknown
        self.threshold = threshold
        self.batch_size = MODELS_BATCH_SIZE.get(model_name, 32)
        self.lr = LR
        self.epochs = NO_EPOCHS
        self.hidden_size = hidden_size
        self.just_three = just_three
        self.model_name = model_name
        self.random_split = random_split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.labeled = [i for i in instances if i.is_known()]
        self.unknown = [i for i in instances if not i.is_known()]

        self.model = CodeBERTClassifier()
        self.model.to(self.device)

    def get_tokens(self, insts):
        
        token_ids_vec = [i.get_tokens() for i in insts]
        masks_vec = [i.get_masks() for i in insts]

        return np.array(token_ids_vec, dtype=np.long), np.array(masks_vec, dtype=np.long)

    def run(self):

        if self.random_split:
            folds = split_folds(self.labeled, self.unknown, self.train_with_unknown)
        else:
            if self.just_three:
                n_split = 4
            else:
                n_split = 3
            folds = split_folds_programs(self.instances, self.train_with_unknown, n_splits=n_split)

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

            train_token_ids, train_masks = self.get_tokens(train)
            y_train = np.array([1 if i.get_label() else 0 for i in train])

            test_token_ids, test_masks = self.get_tokens(test)
            y_test = np.array([1 if i.get_label() else 0 for i in test])

            # Create loaders
            train_loader = self.create_loader(train_token_ids, train_masks, y_train, train, shuffle=True)
            test_loader = self.create_loader(test_token_ids, test_masks, y_test, test, shuffle=False)

            # Reinit model and optimizer
            # Re-initialize the model for each fold
            self.model = CodeBERTClassifier()
            self.model.to(self.device)
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
        
        # saving the instances
        

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

        if not self.raw_baseline:
            if self.train_with_unknown:
                metrics_path = f"{self.output_dir}/nn_{self.threshold}_trained_on_unknown.csv"
            elif self.make_balance:
                metrics_path = f"{self.output_dir}/{self.model_name}_{"random" if self.random_split else "programwise"}_{self.threshold}_trained_on_known_{self.make_balance[0]}_{self.make_balance[1]}.csv"
            else:
                metrics_path = f"{self.output_dir}/{self.model_name}_{"random" if self.random_split else "programwise"}_{self.threshold}_trained_on_known.csv"
        else:
            metrics_path = f"{self.output_dir}/nn_raw_{self.threshold}.csv"

        # write_results_to_csv(all_eval, res_path)
        write_instances_to_file(self.instances, metrics_path.replace('.csv', '.pkl'))
        write_metrics_to_csv(all_metrics, metrics_path)

    def create_loader(self, token_ids, masks, y, instances, shuffle=True):
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        mask_tensor = torch.tensor(masks, dtype=torch.long)
        label_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(token_ids_tensor, mask_tensor, label_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_loop(self, loader, optimizer, loss_fn):
        self.model.train()
        for epoch in range(self.epochs):
            loop = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
            for token_ids, mask, label in loop:
                token_ids, mask, label = token_ids.to(self.device), mask.to(self.device), label.to(self.device)
                output = self.model(token_ids, mask)
                loss = loss_fn(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def eval_loop(self, loader, test_instances):
        self.model.eval()
        all_preds = []
        all_confs = []
        with torch.no_grad():
            for token_ids, mask, _ in loader:
                token_ids, mask = token_ids.to(self.device), mask.to(self.device)
                output = self.model(token_ids, mask)
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

