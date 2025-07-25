import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from spatial_egt.classification.common import df_to_xy, get_feature_data, get_model, read_in_data
from spatial_egt.classification.model_eval_utils import (
    plot_scatter_prob,
    plot_confusion_matrix,
    learning_curve,
    roc_curve,
)
from spatial_egt.common import get_data_path


def cross_val(X, y):
    train_indices = []
    test_indices = []
    y_train_trues = []
    y_test_trues = []
    y_train_probs = []
    y_test_probs = []
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True)
    for train_i, test_i in cross_validation.split(X, y):
        X_train = [X[i] for i in train_i]
        X_test = [X[i] for i in test_i]
        y_train = [y[i] for i in train_i]
        y_test = [y[i] for i in test_i]
        clf = get_model().fit(X_train, y_train)
        y_train_pred = clf.predict_proba(X_train)
        y_test_pred = clf.predict_proba(X_test)
        train_indices.append(train_i)
        test_indices.append(test_i)
        y_train_trues.append(y_train)
        y_test_trues.append(y_test)
        y_train_probs.append(y_train_pred)
        y_test_probs.append(y_test_pred)
    return (train_indices, test_indices, y_train_trues, y_train_probs, y_test_trues, y_test_probs)


def flatten_lists(lists):
    flat_lists = []
    for li in lists:
        flat_lists.append([x for y in li for x in y])
    return flat_lists


def prob_to_pred(y):
    return [np.argmax(sample) for sample in y]


def result_to_dataframe(data_type, label_name, all_df, indices, y_trues, y_probs, y_preds):
    k = [[i for _ in range(len(indices[i]))] for i in range(len(indices))]
    k, indices, y_trues, y_probs, y_preds = flatten_lists([k, indices, y_trues, y_probs, y_preds])
    y_probs_split = {f"{i}_prob": [x[i] for x in y_probs] for i in range(4)}
    true_probs = [y_probs[i][y_trues[i]] for i in range(len(y_trues))]
    data = {"k": k, "true": y_trues, "pred": y_preds, "true_prob": true_probs}
    df = pd.DataFrame(data | y_probs_split, index=indices)
    df["correct"] = df["true"] == df["pred"]

    df = df.merge(all_df, left_index=True, right_index=True)
    df["sample"] = df["sample"].astype(str)
    df = df.set_index(["source", "sample"], drop=True)
    data_path = get_data_path(data_type, ".")
    df_labels = pd.read_csv(f"{data_path}/labels.csv")
    df_labels["sample"] = df_labels["sample"].astype(str)
    df_labels = df_labels.set_index(["source", "sample"], drop=True)
    df_labels = df_labels.drop([label_name], axis=1)
    df = df.merge(df_labels, left_index=True, right_index=True)

    df["C-A"] = df["c"] - df["a"]
    df["B-D"] = df["b"] - df["d"]
    df["Stationary Solution"] = (df["c"] - df["a"]) / ((df["c"] - df["a"]) + (df["b"] - df["d"]))
    df.loc[df["game"] == "Sensitive Wins", "Stationary Solution"] = 0
    df.loc[df["game"] == "Resistant Wins", "Stationary Solution"] = 1

    return df


def main(args):
    data_type, time, label_name, feature_names = read_in_data(args)
    save_loc, df, feature_names = get_feature_data(
        data_type, time, label_name, feature_names
    )
    feature_df = df[feature_names + [label_name]]
    X, y, int_to_class = df_to_xy(feature_df, feature_names, label_name)

    cross_val_results = cross_val(X, y)
    train_indices, test_indices = cross_val_results[0:2]
    y_train_trues, y_train_probs = cross_val_results[2:4]
    y_test_trues, y_test_probs = cross_val_results[4:6]
    y_train_preds = [prob_to_pred(probs) for probs in y_train_probs]
    y_test_preds = [prob_to_pred(probs) for probs in y_test_probs]

    plot_confusion_matrix(save_loc, "train", int_to_class, y_train_trues, y_train_preds)
    plot_confusion_matrix(save_loc, "test", int_to_class, y_test_trues, y_test_preds)
    roc_curve(save_loc, label_name, "test", int_to_class, y_test_trues, y_test_probs)

    if label_name == "game":
        df_test = result_to_dataframe(
            data_type, label_name, df, test_indices, y_test_trues, y_test_probs, y_test_preds
        )
        plot_scatter_prob(save_loc, "test", df_test, "C-A", "B-D", "true_prob")
        plot_scatter_prob(save_loc, "test", df_test, "C-A", "B-D", "correct")
        plot_scatter_prob(save_loc, "test", df_test, "C-A", "B-D", "0_prob")
        plot_scatter_prob(save_loc, "test", df_test, "initial_fs", "initial_density", "correct")

    learning_curve(save_loc, X, y)


if __name__ == "__main__":
    main(sys.argv)
