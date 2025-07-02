import sys

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spatial_egt.classification.common import get_feature_data

import warnings
warnings.filterwarnings("ignore")


def run(X, y):
    estimator = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, solver="adam")
    clf = make_pipeline(StandardScaler(), estimator)
    scores = cross_val_score(clf, X, y, cv=5)
    return np.mean(scores)


def sfs(save_loc, feature_df, feature_names, num_features_start):
    #turn game data into classes readable by ML model
    class_to_int = {lc:i for i,lc in enumerate(feature_df["game"].unique())}
    y = [class_to_int[x] for x in feature_df["game"].values]

    #get sets of features already evaluated and keep only the top 10
    if num_features_start == "0":
        starting_sets = [[]]
    else:
        previous_results = open(f"{save_loc}/{num_features_start}.csv", encoding="UTF-8").read()
        previous_results = [x.split(",") for x in previous_results.split("\n")]
        top_n = 10 if len(previous_results) > 10 else len(previous_results)
        starting_sets = [x[:-1] for x in previous_results[1:top_n+1]]

    #add each feature to the top 10 best of the given size
    results = dict()
    for set_of_features in starting_sets:
        for feature_name in feature_names:
            if feature_name in set_of_features:
                continue
            full_feature_set_list = set_of_features + [feature_name]
            full_feature_set = tuple(sorted(full_feature_set_list))
            if full_feature_set in results:
                continue
            X = list(feature_df[full_feature_set_list].values)
            mean = run(X, y)
            results[full_feature_set] = mean

    #save the new feature sets and scores
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    num_features = int(num_features_start)+1
    with open(f"{save_loc}/{num_features}.csv", "w", encoding="UTF-8") as f:
        f.write(",".join([str(i) for i in range(num_features)])+","+"Mean Accuracy\n")
        for feature_set, mean in results.items():
            f.write(",".join(feature_set)+f",{mean:5.3f}\n")


def main(data_type, label_name, num_features_start, feature_names):
    save_loc, df, feature_names = get_feature_data(data_type, label_name, feature_names, "sfs")
    feature_df = df[feature_names+[label_name]]
    sfs(save_loc, feature_df, feature_names, num_features_start)


if __name__ == "__main__":
    if len(sys.argv) > 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        print("Please provide the data type, label name, number of features, and feature set/names.")
