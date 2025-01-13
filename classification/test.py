import pickle
import sys

import pandas as pd

from common.common import get_data_path
from common.classification import (clean_feature_data, df_to_xy,
                                   features, plot_confusion_matrix)


def test_model(save_loc, X, y, int_to_name):
    with open(f"{save_loc}/model.pkl", "rb") as f:
        clf = pickle.load(f)
    disp_labels = [int_to_name[x] for x in clf.classes_]
    y_pred = clf.predict(X)
    acc = sum([y_pred[i] == y[i] for i in range(len(y))])/len(y)
    plot_confusion_matrix(save_loc, "confusion_test", disp_labels, y, y_pred, acc)


def main(*data_types):
    save_loc = get_data_path(".", "model")

    df = pd.DataFrame()
    for data_type in data_types[0]:
        features_data_path = get_data_path(data_type, "features")
        df_dt = pd.read_csv(f"{features_data_path}/all.csv")
        df = pd.concat([df, df_dt])
    df = clean_feature_data(df)

    label = ["game"]
    if len(features) == 0:
        feature_df = df
    else:
        feature_df = df[features+label]
    X, y, int_to_name = df_to_xy(feature_df)
    
    test_model(save_loc, X, y, int_to_name)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print("Please provide the data types to test the model with.")