import pickle
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


def test_model(save_loc, df, label_name):
    feature_names = list(df.columns)
    feature_names.remove(label_name)
    label_categories = df[label_name].unique()
    category_to_int = {lc:i for i,lc in enumerate(label_categories)}
    int_to_category = {i:lc for i,lc in enumerate(label_categories)}
    X = list(df[feature_names].values)
    y = [category_to_int[x] for x in df[label_name].values]

    with open(f"{save_loc}/model.pkl", "rb") as f:
        clf = pickle.load(f)
    y_pred = clf.predict(X)
    acc = round(sum([y_pred[i] == y[i] for i in range(len(y))])/len(y), 2)
    plot_confusion_matrix(save_loc, int_to_category, clf, y, y_pred, acc, "")


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
    
    test_model(data_path, feature_df, label[0])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print("Please provide the data types to test the model with.")