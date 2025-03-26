import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classification.common import read_and_clean_features
from common import get_data_path

import warnings
warnings.filterwarnings("ignore")


def run(X, y):
    estimator = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, solver="lbfgs")
    clf = make_pipeline(StandardScaler(), estimator)
    scores = cross_val_score(clf, X, y, cv=5)
    return np.mean(scores)


def main(experiment_name, num_features_start, *data_types):
    #read in and clean feature data
    parent_dir = "."
    if len(data_types[0]) == 1:
        parent_dir = data_types[0][0]
    save_loc = get_data_path(parent_dir, f"model/{experiment_name}/model_iteration")
    feature_df = read_and_clean_features(data_types[0], ["game"], experiment_name)
    feature_names = list(feature_df.columns)
    feature_names.remove("game")

    #turn game data into classes readable by ML model
    class_to_int = {lc:i for i,lc in enumerate(feature_df["game"].unique())}
    y = [class_to_int[x] for x in feature_df["game"].values]

    #get sets of features already evaluated and keep only the top 10
    if num_features_start == "0":
        starting_sets = [[]]
    else:
        data_path = get_data_path(parent_dir, f"model/{experiment_name}/model_iteration")
        df_start = pd.read_csv(f"{data_path}/{num_features_start}.csv")
        df_start_top = df_start.nlargest(10, "value")
        cols = [x for x in df_start_top.columns if x != "value"]
        starting_sets = df_start_top[cols].values.tolist()

    #add each feature to the top 10 best of the given size
    results = []
    for set_of_features in starting_sets:
        for feature_name in feature_names:
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


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        print("Please provide a feature set, staring number of features, and the data types to train the model with.")
