import numpy as np
import pandas as pd
from scipy.sparse import csgraph, csr_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spatial_egt.common import game_colors, get_data_path


def df_to_xy(df, feature_names, label_name):
    if label_name == "game":
        label_classes = list(game_colors.keys())
    else:
        label_classes = list(df[label_name].unique())
    class_to_int = {lc:i for i,lc in enumerate(label_classes)}
    int_to_class = {i:lc for i,lc in enumerate(label_classes)}
    X = df[feature_names].values.tolist()
    y = [class_to_int[x] for x in df[label_name].values]
    return X, y, int_to_class


def remove_correlated(df, feature_names):
    corr_matrix = df[feature_names].corr(method="spearman")
    high_corr = (corr_matrix >= 0.9) | (corr_matrix <= -0.9)

    adj_matrix = csr_matrix(high_corr)
    _, labels = csgraph.connected_components(csgraph=adj_matrix, directed=False)
    clusters = [[] for _ in range(len(set(labels)))]
    for i, label in enumerate(labels):
        clusters[label].append(feature_names[i])
    features_to_keep = ["Proportion_Sensitive" if "Proportion_Sensitive" in x else x[0] for x in clusters]

    return features_to_keep


def feature_set_to_names(df, feature_names, label_name):
    feature_df = df.drop(["source", "sample"], axis=1)
    all_feature_names = list(feature_df.drop(label_name, axis=1).columns)
    if feature_names == ["all"]:
        true_features = all_feature_names
    elif feature_names == ["noncorr"]:
        true_features = remove_correlated(feature_df, all_feature_names)
    else:
        true_features = feature_names
    return true_features


def read_and_clean_feature_df(data_type, time):
    features_data_path = get_data_path(data_type, "statistics", time)
    df = pd.read_csv(f"{features_data_path}/features.csv")
    #df = df.fillna(0)
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], 0)
    return df


def get_feature_data(data_type, time, label_name, feature_names, extra_dir=""):
    data_dir = "_".join(feature_names)
    save_loc = get_data_path(data_type, f"images/model/{data_dir}/{extra_dir}", time)
    df = read_and_clean_feature_df(data_type, time)
    feature_names = feature_set_to_names(df, feature_names, label_name)
    return save_loc, df, feature_names


def read_in_data(args):
    if len(args) < 5:
        print("Please provide the data type, time, label name, and feature set/names.")
        exit()
    data_type = args[1]
    time = args[2]
    label_name = args[3]
    feature_names = args[4:]
    return data_type, time, label_name, feature_names


def get_model():
    estimator = MLPClassifier(hidden_layer_sizes=(300,), max_iter=5000, solver="adam")
    clf = make_pipeline(StandardScaler(), estimator)
    return clf
