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
    estimator = MLPClassifier(hidden_layer_sizes=(1,))
    clf = make_pipeline(StandardScaler(), estimator)
    scores = cross_val_score(clf, X, y, cv=5)
    return np.mean(scores)


def main(experiment_name, num_features, *data_types):
    parent_dir = "."
    if len(data_types[0]) == 1:
        parent_dir = data_types[0][0]
    save_loc = get_data_path(parent_dir, f"model/{experiment_name}/model_iteration")
    feature_df = read_and_clean_features(data_types[0], ["game"], experiment_name)
    feature_names = list(feature_df.columns)
    feature_names.remove("game")

    class_to_int = {lc:i for i,lc in enumerate(feature_df["game"].unique())}
    y = [class_to_int[x] for x in feature_df["game"].values]

    results = []
    num_features = int(num_features)
    features = combinations(feature_names, num_features)
    for set_of_features in features:
        set_of_features = list(set_of_features)
        X = list(feature_df[set_of_features].values)
        mean = run(X, y)
        set_of_features.append(f"{mean:5.3f}")
        results.append(set_of_features)

    with open(f"{save_loc}/{num_features}.csv", "w") as f:
        f.write(",".join([str(i) for i in range(num_features)])+","+"mean\n")
        for result in results:
            f.write(",".join(result)+"\n")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        print("Please provide a feature set, number of features, and the data types to train the model with.")
