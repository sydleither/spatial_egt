import pickle
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from common.common import get_data_path
from common.classification import (clean_feature_data, df_to_xy,
                                   features, plot_confusion_matrix)


def train_model(X_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=(500,250,100,50)).fit(X_train, y_train)
    return clf


def cross_val(save_loc, X, y, int_to_name):
    n_splits = 5
    cross_validation = StratifiedKFold(n_splits=n_splits, shuffle=True)
    avg_acc = 0
    for k, (train_i, test_i) in enumerate(cross_validation.split(X, y)):
        X_train = [X[i] for i in train_i]
        X_test = [X[i] for i in test_i]
        y_train = [y[i] for i in train_i]
        y_test = [y[i] for i in test_i]
        clf = train_model(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = sum([y_pred[i] == y_test[i] for i in range(len(y_test))])/len(y_test)
        avg_acc += acc
        disp_labels = [int_to_name[x] for x in clf.classes_]
        plot_confusion_matrix(save_loc, f"confusion_{k}", disp_labels, y_test, y_pred, acc)
    print(f"Average Accuracy: {avg_acc/n_splits}")


def save_model(save_loc, X, y):
    clf = train_model(X, y)
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
    #save_model(save_loc, X, y)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print("Please provide the data types to train the model with.")