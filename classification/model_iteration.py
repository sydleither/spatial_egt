from itertools import combinations
import sys

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classification.common import read_and_clean_features
from common import get_data_path

import warnings
warnings.filterwarnings("ignore")


def run(X, y):
    estimator = MLPClassifier(hidden_layer_sizes=(100,))
    clf = make_pipeline(StandardScaler(), estimator)
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)
    return np.mean(scores), np.std(scores)


def main(experiment_name, *data_types):
    parent_dir = "."
    if len(data_types[0]) == 1:
        parent_dir = data_types[0][0]
    save_loc = get_data_path(parent_dir, f"model/{experiment_name}")
    feature_df = read_and_clean_features(data_types[0], ["game"], experiment_name)
    feature_names = list(feature_df.columns)
    feature_names.remove("game")
    
    class_to_int = {lc:i for i,lc in enumerate(feature_df["game"].unique())}
    for num_features in range(5, 11):
        features = combinations(feature_names, num_features)
        for set_of_features in features:
            X = list(feature_df[feature_names].values)
            y = [class_to_int[x] for x in feature_df["game"].values]
            mean, std = run(X, y)
            print(set_of_features)
            print(mean, std)
            print()


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide a feature set and the data types to train the model with.")
