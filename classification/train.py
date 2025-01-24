import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from common.common import get_data_path
from common.classification import (clean_feature_data, df_to_xy,
                                   features, plot_confusion_matrix,
                                   plot_performance_stats,
                                   plot_prediction_distributions)


def train_model(X, y):
    clf = MLPClassifier(hidden_layer_sizes=(500,250,100,50)).fit(X, y)
    #clf = DecisionTreeClassifier(max_depth=10).fit(X, y)
    #clf = RandomForestClassifier().fit(X, y)
    return clf


def cross_val(save_loc, X, y, int_to_name):
    all_y_train = []
    all_y_test = []
    all_y_train_pred = []
    all_y_test_pred = []
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True)
    for k, (train_i, test_i) in enumerate(cross_validation.split(X, y)):
        X_train = [X[i] for i in train_i]
        X_test = [X[i] for i in test_i]
        y_train = [y[i] for i in train_i]
        y_test = [y[i] for i in test_i]
        clf = train_model(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        plot_confusion_matrix(save_loc, f"confusion_train_{k}", int_to_name, y_train, y_train_pred)
        plot_confusion_matrix(save_loc, f"confusion_test_{k}", int_to_name, y_test, y_test_pred)
        plot_performance_stats(save_loc, f"stats_train_{k}", int_to_name, y_train, y_train_pred)
        plot_performance_stats(save_loc, f"stats_test_{k}", int_to_name, y_test, y_test_pred)
        all_y_train += y_train
        all_y_test += y_test
        all_y_train_pred += y_train_pred.tolist()
        all_y_test_pred += y_test_pred.tolist()
    plot_confusion_matrix(save_loc, "confusion_train_all", int_to_name, all_y_train, all_y_train_pred)
    plot_confusion_matrix(save_loc, "confusion_test_all", int_to_name, all_y_test, all_y_test_pred)
    plot_performance_stats(save_loc, "stats_train_all", int_to_name, all_y_train, all_y_train_pred)
    plot_performance_stats(save_loc, "stats_test_all", int_to_name, all_y_test, all_y_test_pred)


def save_model(save_loc, X, y, int_to_name):
    clf = train_model(X, y)
    y_pred = clf.predict(X)
    plot_confusion_matrix(save_loc, f"confusion_train", int_to_name, y, y_pred)
    plot_performance_stats(save_loc, f"stats_train", int_to_name, y, y_pred)
    with open(f"{save_loc}/model.pkl", "wb") as f:
        pickle.dump(clf, f)


def main(experiment_name, *data_types):
    save_loc = get_data_path(".", f"model/{experiment_name}")

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
    
    cross_val(save_loc, X, y, int_to_name)
    #save_model(save_loc, X, y, int_to_name)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide an experiment name and the data types to train the model with.")