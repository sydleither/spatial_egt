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
                                   plot_prediction_distributions)


def train_model(X, y):
    clf = MLPClassifier(hidden_layer_sizes=(500,250,100,50)).fit(X, y)
    #clf = DecisionTreeClassifier(max_depth=10).fit(X, y)
    #clf = RandomForestClassifier().fit(X, y)
    return clf


def cross_val(save_loc, X, y, int_to_name):
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True)
    train_accs = []
    test_accs = []
    for k, (train_i, test_i) in enumerate(cross_validation.split(X, y)):
        X_train = [X[i] for i in train_i]
        X_test = [X[i] for i in test_i]
        y_train = [y[i] for i in train_i]
        y_test = [y[i] for i in test_i]
        clf = train_model(X_train, y_train)
        disp_labels = [int_to_name[x] for x in clf.classes_]
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        train_acc = sum([y_train_pred[i] == y_train[i] for i in range(len(y_train))])/len(y_train)
        test_acc = sum([y_test_pred[i] == y_test[i] for i in range(len(y_test))])/len(y_test)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        plot_confusion_matrix(save_loc, f"confusion_train_{k}", disp_labels, y_train, y_train_pred, train_acc)
        plot_confusion_matrix(save_loc, f"confusion_test_{k}", disp_labels, y_test, y_test_pred, test_acc)
    print(f"Train: {np.mean(train_accs):5.3f}, {np.std(train_accs):5.3f}")
    print(f"Test: {np.mean(test_accs):5.3f}, {np.std(test_accs):5.3f}")


def save_model(save_loc, X, y, int_to_name):
    clf = train_model(X, y)
    disp_labels = [int_to_name[x] for x in clf.classes_]
    y_pred = clf.predict(X)
    acc = sum([y_pred[i] == y[i] for i in range(len(y))])/len(y)
    plot_confusion_matrix(save_loc, f"confusion_train", disp_labels, y, y_pred, acc)
    plot_prediction_distributions(save_loc, X, features, y, y_pred, disp_labels, features)
    with open(f"{save_loc}/model.pkl", "wb") as f:
        pickle.dump(clf, f)


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
    
    cross_val(save_loc, X, y, int_to_name)
    #save_model(save_loc, X, y, int_to_name)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print("Please provide the data types to train the model with.")