import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

class TraceClassifier:
    def __init__(self, feature_file):
        self.feature_file = feature_file
        self.model = None
    
    def load_data(self):
        """Loads feature vectors and labels from a CSV file."""
        df = pd.read_csv(self.feature_file)
        X = df.drop(columns=["filename", "label"], errors='ignore').values  # Feature matrix
        y = df["label"].values if "label" in df.columns else None  # Labels (if available)
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
    
if __name__ == "__main__":
    feature_file = "features.csv"  # Ensure this contains a 'label' column
    classifier = TraceClassifier(feature_file)
    
    X, y = classifier.load_data()
    X_train, X_test, y_train, y_test = classifier.preprocess_data(X, y)
    
    classifier.train_model(X_train, y_train, model_type="mlp")  # Choose "random_forest" or "mlp"
    classifier.evaluate_model(X_test, y_test)
