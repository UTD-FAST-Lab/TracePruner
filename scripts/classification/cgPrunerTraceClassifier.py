import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

import os

class TraceClassifier:
    def __init__(self, feature_file, cgpruner_file, results_dir):
        self.feature_file = feature_file
        self.cgpruner_file = cgpruner_file
        self.model = None
        self.results_dir = results_dir
    
    def load_data(self):
        """Loads feature vectors and labels from a CSV file, merging with cgpruner data."""
        # Load both datasets
        df_features = pd.read_csv(self.feature_file)
        df_cgpruner = pd.read_csv(self.cgpruner_file)

            # Drop unwanted columns from cgpruner
        columns_to_exclude = ["wiretap", "wala-cge-0cfa-noreflect-intf-direct", "wala-cge-0cfa-noreflect-intf-trans", "offset"]
        df_cgpruner_filtered = df_cgpruner.drop(columns=columns_to_exclude, errors='ignore')

        # Merge on 'program_name', 'method', and 'target'
        df_features_unique = df_features.drop_duplicates(subset=['program_name', 'method', 'target'], keep='last')
        df_cgpruner_unique = df_cgpruner_filtered.drop_duplicates(subset=['program_name', 'method', 'target'])
        df_merged = df_features_unique.merge(df_cgpruner_unique, on=['program_name', 'method', 'target'], how='left')


        # Drop non-feature columns and prepare X, y
        X = df_merged.drop(columns=["program_name", "label", 'edge_name', 'method', 'target'], errors='ignore').values  
        y = df_merged["label"].values if "label" in df_merged.columns else None  

        return X, y
    
    def preprocess_data(self, X, y=None):
        """Normalizes the feature data and splits into train/test sets."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        return X_scaled
    
    def train_model(self, X_train, y_train, model_type="random_forest"):
        """Trains a classifier (Random Forest or MLP)."""
        if model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "mlp":
            self.model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        else:
            raise ValueError("Unsupported model type")
        
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluates the classifier and prints the accuracy and classification report."""
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", acc)
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def checkoverfitting(self, X_train, X_test, y_train, y_test):

        # Get predictions for both training and test sets
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate and print training and test accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Print detailed classification report for the test set
        rf_report = classification_report(y_test, y_test_pred)
        print("Random Forest Classification Report:")
        print(rf_report)

        # --- Plot the Learning Curve ---
        # We'll use the entire dataset (X and y) with 5-fold cross-validation.
        train_sizes, train_scores, valid_scores = learning_curve(
            estimator=self.model,
            X=X,
            y=y,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        # Compute average scores for plotting
        train_scores_mean = np.mean(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
        plt.plot(train_sizes, valid_scores_mean, 'o-', color='g', label='Cross-Validation Score')
        plt.title("Learning Curve for Random Forest")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(os.path.join(self.results_dir, "w2v-cgp.png"), dpi=300)  # Saves the plot to a file
        plt.close()  # Closes the figure
    
if __name__ == "__main__":

    dataset_dir = '/home/mohammad/projects/CallGraphPruner/data/datasets/branches'

    # feature_file = os.path.join(dataset_dir, 'combined_lstm_dataset_v0.csv')  # Ensure this contains a 'label' column
    # feature_file = os.path.join(dataset_dir, 'combined_mixed_dataset_v0.csv')  # Ensure this contains a 'label' column
    feature_file = os.path.join(dataset_dir, 'combined_w2v_dataset_v0.csv')  # Ensure this contains a 'label' column
    cgpruner_file = os.path.join(dataset_dir, 'combined_w2v_dataset_v0_cgpruner.csv')

    results_dir = '/home/mohammad/projects/CallGraphPruner/data/results/branches' 

    classifier = TraceClassifier(feature_file, cgpruner_file, results_dir)
    
    X, y = classifier.load_data()
    X_train, X_test, y_train, y_test = classifier.preprocess_data(X, y)
    
    classifier.train_model(X_train, y_train, model_type="random_forest")  # Choose "random_forest" or "mlp"
    # classifier.evaluate_model(X_test, y_test)
    classifier.checkoverfitting(X_train, X_test, y_train, y_test)
