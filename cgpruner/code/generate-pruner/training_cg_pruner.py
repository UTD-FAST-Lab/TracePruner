# mohammad rafieian


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

data_folder = '/20TB/mohammad'
model = 'original'   #original, config_trace, config


# Path to your training data CSV
TRAINING_DATA_FILE = f'{data_folder}/models/pruner_{model}/training_data/final_training_data.csv'
LEARNED_MODEL_FILE = f'{data_folder}/models/pruner_{model}/pruner_{model}.pkl'

def main():
    # Load the training data
    training_data = pd.read_csv(TRAINING_DATA_FILE)
    
    # Drop unwanted columns
    ignore_columns = ['method', 'target', 'offset', '#edge_disjoint_paths_from_main', '#node_disjoint_paths_from_main']
    training_data = training_data.drop(columns=[col for col in ignore_columns if col in training_data.columns])

    # Separate the features (X) and the label (y)
    X = training_data.drop(columns=['wiretap'])  # All features except the label
    y = training_data['wiretap']  # The target label

    # Split the data into training and testing sets (optional, for validation purposes)
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialize the RandomForestClassifier with the given parameters
    clf = RandomForestClassifier(
        n_estimators=1000,
        max_features="sqrt",
        random_state=0,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=False,
        criterion="entropy"
    )

    # Train the classifier
    # clf = clf.fit(x_train, y_train)
    clf = clf.fit(X, y)
    
    # Save the trained model to a file
    joblib.dump(clf, LEARNED_MODEL_FILE)
    print("Training Complete")

if __name__ == "__main__":
    main()
