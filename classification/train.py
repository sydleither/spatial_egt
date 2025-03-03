import pickle
import sys

from classification.common import df_to_xy, get_model, read_and_clean_features
from classification.performance_plots import plot_all
from common import get_data_path


def save_model(save_loc, X, y, int_to_name):
    clf = get_model().fit(X, y)
    y_pred = clf.predict(X)
    plot_all(save_loc, int_to_name, [y], [y_pred], "train")
    with open(f"{save_loc}/model.pkl", "wb") as f:
        pickle.dump(clf, f)


def main(experiment_name, *data_types):
    parent_dir = "."
    if len(data_types[0]) == 1:
        parent_dir = data_types[0][0]
    save_loc = get_data_path(parent_dir, f"model/{experiment_name}")
    feature_df = read_and_clean_features(data_types[0], ["game"], experiment_name)
    X, y, int_to_name, _ = df_to_xy(feature_df, "game")
    save_model(save_loc, X, y, int_to_name)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide a feature set and the data types to train the model with.")
