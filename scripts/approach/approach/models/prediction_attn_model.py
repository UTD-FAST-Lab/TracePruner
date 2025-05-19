import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn.init as init


class ConfidenceWeightedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super(ConfidenceWeightedNN, self).__init__()
        layers = []
        for _ in range(num_layers - 1):
            layer = nn.Linear(input_dim, hidden_dim)
            # init.kaiming_normal_(layer.weight)  # He initialization
            layers.append(layer)
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        final_layer = nn.Linear(hidden_dim, 1)
        # init.kaiming_normal_(final_layer.weight)  # He initialization
        layers.append(final_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Check for NaNs in the input
        if torch.isnan(x).any():
            print("NaN detected in input!")
            print(x)
            exit(1)

        # Forward pass through the layers
        for i, layer in enumerate(self.model):
            x = layer(x)
            if torch.isnan(x).any():
                print(f"NaN detected in layer {i} ({layer})!")
                print(x)
                exit(1)

        # Sigmoid for binary classification
        x = torch.sigmoid(x)

        # Check for NaNs in the final output
        if torch.isnan(x).any():
            print("NaN detected in final output!")
            print(x)
            exit(1)

        return x.squeeze()


def confidence_weighted_loss(preds, targets, confidences, epsilon=1e-8):
    # Binary cross-entropy with confidence scaling
    preds = torch.clamp(preds, epsilon, 1 - epsilon)
    loss = -confidences * (targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
    return loss.mean()


def standard_binary_cross_entropy(preds, targets, epsilon=1e-8):
    # Clamp to avoid log(0)
    preds = torch.clamp(preds, epsilon, 1 - epsilon)
    loss = -(targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
    return loss.mean()


class InstanceDataset(Dataset):
    def __init__(self, instances, feature_type='static'):
        self.instances = instances
        self.feature_type = feature_type

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        # Select the feature type
        if self.feature_type == 'static':
            features = instance.get_static_featuers()
        elif self.feature_type == 'trace':
            features = instance.get_trace_features()
            if features is None:
                features = [0.0] * 128
        elif self.feature_type == 'semantic':
            features = instance.get_semantic_features()
            if features is None:
                features = [0.0] * 768  # Default to zero vector if None
        elif self.feature_type == 'var':
            features = instance.get_var_features()
            if features is None:
                features = [0.0] * 64
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
        
        if instance.get_cluster_label() is None:
            # If the instance is not labeled, assign a default label
            label = torch.tensor(0, dtype=torch.float32)
            confidence = torch.tensor(0, dtype=torch.float32)
        else:
            label = torch.tensor(instance.get_cluster_label(), dtype=torch.float32)
            confidence = torch.tensor(instance.get_confidence(), dtype=torch.float32)
        return torch.tensor(features, dtype=torch.float32), label, confidence, instance


def train_model(model, train_loader, optimizer, num_epochs=10, device='cuda', apply_attention=True):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for features, labels, confidences, _ in train_loader:
            features, labels, confidences = features.to(device), labels.to(device), confidences.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            if apply_attention:
                loss = confidence_weighted_loss(outputs, labels, confidences)
            else:
                loss = standard_binary_cross_entropy(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")


def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels, _, instances in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = outputs.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            # Store the predictions in the instances
            for instance, pred in zip(instances, preds):
                instance.set_predicted_label(int(pred > 0.5))
    return all_preds, all_labels
