import pickle
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from common.common import get_data_path
from common.features import clean_feature_data, features


def plot_confusion_matrix(save_loc, int_to_category, clf, y_test, y_pred, acc, k):
    conf_mat = confusion_matrix(y_test, y_pred, normalize="true")
    disp_labels = [int_to_category[x] for x in clf.classes_]

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=disp_labels)
    disp.plot(ax=ax)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    ax.set_title(f"Accuracy: {acc}")
    fig.tight_layout()
    fig.savefig(f"{save_loc}/confusion_matrix{k}.png", bbox_inches="tight", transparent=True)


def cross_val(save_loc, df, label_name):
    feature_names = list(df.columns)
    feature_names.remove(label_name)
    label_categories = df[label_name].unique()
    category_to_int = {lc:i for i,lc in enumerate(label_categories)}
    int_to_category = {i:lc for i,lc in enumerate(label_categories)}
    X = list(df[feature_names].values)
    y = [category_to_int[x] for x in df[label_name].values]

    avg_acc = 0
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True)
    for k, (train_i, test_i) in enumerate(cross_validation.split(X, y)):
        X_train = [X[i] for i in train_i]
        X_test = [X[i] for i in test_i]
        y_train = [y[i] for i in train_i]
        y_test = [y[i] for i in test_i]
        clf = MLPClassifier(hidden_layer_sizes=(500,250,100,50)).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = round(sum([y_pred[i] == y_test[i] for i in range(len(y_test))])/len(y_test), 2)
        avg_acc += acc
        plot_confusion_matrix(save_loc, int_to_category, clf, y_test, y_pred, acc, k)
        print(f"\tAccuracy {k}: {acc}")
    print("\tAverage Accuracy:", avg_acc/5)


def save_model(save_loc, df, label_name):
    feature_names = list(df.columns)
    feature_names.remove(label_name)
    label_categories = df[label_name].unique()
    category_to_int = {lc:i for i,lc in enumerate(label_categories)}
    int_to_category = {i:lc for i,lc in enumerate(label_categories)}
    X = list(df[feature_names].values)
    y = [category_to_int[x] for x in df[label_name].values]

    clf = MLPClassifier(hidden_layer_sizes=(500,250,100,50)).fit(X, y)
    with open(f"{save_loc}/model.pkl", "wb") as f:
        pickle.dump(clf, f)


def main(*data_types):
    data_path = get_data_path(".", ".")

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
    
    #cross_val(data_path, feature_df, label[0])
    save_model(data_path, feature_df, label[0])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print("Please provide the data types to train the model with.")