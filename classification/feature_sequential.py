import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classification.common import get_feature_data

import warnings
warnings.filterwarnings("ignore")


def run(X, y):
    estimator = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, solver="lbfgs")
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
        df_start = pd.read_csv(f"{save_loc}/{num_features_start}.csv")
        df_start_top = df_start.nlargest(10, "value")
        cols = [x for x in df_start_top.columns if x != "value"]
        starting_sets = df_start_top[cols].values.tolist()

    #add each feature to the top 10 best of the given size
    results = []
    for set_of_features in starting_sets:
        for feature_name in feature_names:
            if feature_name in set_of_features:
                continue
            full_feature_set = set_of_features + [feature_name]
            X = list(feature_df[full_feature_set].values)
            mean = run(X, y)
            full_feature_set.append(f"{mean:5.3f}")
            results.append(full_feature_set)

    #save the new feature sets and scores
    num_features = int(num_features_start)+1
    with open(f"{save_loc}/{num_features}.csv", "w") as f:
        f.write(",".join([str(i) for i in range(num_features)])+","+"value\n")
        for result in results:
            f.write(",".join(result)+"\n")


def main(data_type, num_features_start, feature_names):
    save_loc, df, feature_names, label_name = get_feature_data(data_type, feature_names, "model_iteration")
    feature_df = df[feature_names+[label_name]]
    sfs(save_loc, feature_df, feature_names, num_features_start)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        print("Please provide the data type, number of features, and feature set/names.")
