import pandas as pd
from scipy.sparse import csgraph, csr_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import game_colors, get_data_path


def df_to_xy(df, feature_names, label_name):
    label_classes = list(game_colors.keys())
    class_to_int = {lc:i for i,lc in enumerate(label_classes)}
    int_to_class = {i:lc for i,lc in enumerate(label_classes)}
    X = df[feature_names].values.tolist()
    y = [class_to_int[x] for x in df[label_name].values]
    return X, y, int_to_class


def remove_correlated(df, feature_names):
    corr_matrix = df[feature_names].corr(method="spearman")
    high_corr = ((corr_matrix >= 0.9) | (corr_matrix <= -0.9)) & (corr_matrix != 1.0)

    adj_matrix = csr_matrix(high_corr)
    _, labels = csgraph.connected_components(csgraph=adj_matrix, directed=False)
    clusters = [[] for _ in range(len(set(labels)))]
    for i, label in enumerate(labels):
        clusters[label].append(feature_names[i])
    features_to_keep = [x[0] for x in clusters]

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


def read_and_clean_feature_df(data_type, label_name):
    features_data_path = get_data_path(data_type, "features")
    df = pd.read_csv(f"{features_data_path}/all.csv")
    df = df[df[label_name] != "Unknown"]
    len_df = len(df)
    df = df.dropna()
    if len_df != len(df):
        print(f"WARNING: {len_df-len(df)} rows with NA dropped.")
    return df


def get_feature_data(data_type, feature_names):
    label_name = "game"
    data_dir = "_".join(feature_names)
    save_loc = get_data_path(data_type, f"images/model/{data_dir}/features")
    df = read_and_clean_feature_df(data_type, label_name)
    feature_names = feature_set_to_names(df, feature_names, label_name)
    return save_loc, df, feature_names, label_name


def get_model():
    estimator = MLPClassifier(hidden_layer_sizes=(30,), max_iter=5000, solver="adam")
    clf = make_pipeline(StandardScaler(), estimator)
    return clf
