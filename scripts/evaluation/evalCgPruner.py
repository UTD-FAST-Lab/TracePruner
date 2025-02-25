import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

# Load the CSV file that has edge names
def load_edge_dataset(edge_csv_path):
    return pd.read_csv(edge_csv_path)


# Load the features CSV file
def load_features_dataset(features_csv_path):
    return pd.read_csv(features_csv_path)



def main(features_csv_path):
  
    # Load the actual features dataset
    features_df = load_features_dataset(features_csv_path)

    # Remove duplicates based on the join keys (if that's desired)
    features_df = features_df.drop_duplicates(subset=['program_name', 'method', 'target'])

    # Prepare X and y; drop non-feature columns from features_df
    X = features_df.drop(columns=['wiretap', 'method', 'offset', 'target',
                                  'wala-cge-0cfa-noreflect-intf-direct', 'wala-cge-0cfa-noreflect-intf-trans',
                                  'program_name'])
    y = features_df['wiretap']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Get predictions for both training and test sets
    y_train_pred = rf_clf.predict(X_train)
    y_test_pred = rf_clf.predict(X_test)

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
        estimator=rf_clf,
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
    plt.savefig("learning_curve.png", dpi=300)  # Saves the plot to a file
    plt.close()  # Closes the figure
    
    # Train a random (dummy) classifier for comparison
    random_clf = DummyClassifier(strategy='uniform', random_state=42)
    random_clf.fit(X_train, y_train)
    y_random_pred = random_clf.predict(X_test)
    
    random_accuracy = accuracy_score(y_test, y_random_pred)
    random_report = classification_report(y_test, y_random_pred)
    
    print(f"Random Classifier Accuracy: {random_accuracy:.4f}")
    print("Random Classifier Classification Report:")
    print(random_report)

if __name__ == "__main__":
  
    features_csv_path = "/home/mohammad/projects/CallGraphPruner/data/datasets/combined_w2v_dataset_v0_cgpruner.csv"   # CSV file with features, label, src, and target columns
    main(features_csv_path)


