import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

class TraceClassifier:
    def __init__(self, feature_file, results_dir):
        self.feature_file = feature_file
        self.results_dir = results_dir
        self.model = None
        self.results_file = os.path.join(results_dir, "classification_results_mlp.txt")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self):
        df = pd.read_csv(self.feature_file)
        df = df.drop_duplicates(subset=['program_name', 'method', 'target'], keep='last')
        X = df.drop(columns=["program_name", "label", 'edge_name', 'method', 'target'], errors='ignore').values
        y = df["label"].values if "label" in df.columns else None
        return X, y
    
    def preprocess_data(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, y_train, model_type):
        if model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "mlp":
            self.model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        else:
            raise ValueError("Unsupported model type")
        
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        with open(self.results_file, "a") as f:
            f.write(f"Final Test Accuracy: {acc:.4f}\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
    
    def check_overfitting(self, X_train, X_val, y_train, y_val):
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        with open(self.results_file, "a") as f:
            f.write(f"Training Accuracy: {train_acc:.4f}\n")
            f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring='accuracy')
        
        with open(self.results_file, "a") as f:
            f.write(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
        
        train_sizes, train_scores, val_scores = learning_curve(self.model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
        
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training Score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='g', label='Validation Score')
        plt.title("Learning Curve")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(os.path.join(self.results_dir, "learning_curve_mlp.png"), dpi=300)
        plt.close()
    
if __name__ == "__main__":
    dataset_dir = '/home/mohammad/projects/CallGraphPruner/data/datasets/cgs-branches'
    # feature_file = os.path.join(dataset_dir, 'combined_w2v_dataset_v0.csv')
    feature_file = os.path.join(dataset_dir, 'combined_128_features.csv')
    results_dir = '/home/mohammad/projects/CallGraphPruner/data/results/cgs-branches'
    
    classifier = TraceClassifier(feature_file, results_dir)
    
    X, y = classifier.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.preprocess_data(X, y)
    
    classifier.train_model(X_train, y_train, model_type="mlp")
    classifier.check_overfitting(X_train, X_val, y_train, y_val)
    classifier.evaluate_model(X_test, y_test)
