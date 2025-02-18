import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import game_colors, get_data_path


features = []
#features = ["nn_mean", "nc_fs_std", "nc_fs_skew", "nc_fr_std", "nc_fr_skew"]


def df_to_xy(df, label_name):
    feature_names = list(df.columns)
    feature_names.remove(label_name)
    label_classes = list(game_colors.keys())
    class_to_int = {lc:i for i,lc in enumerate(label_classes)}
    int_to_class = {i:lc for i,lc in enumerate(label_classes)}
    X = list(df[feature_names].values)
    y = [class_to_int[x] for x in df[label_name].values]
    return X, y, int_to_class, feature_names


def clean_feature_data(df):
    df = df[df["game"] != "unknown"]
    skew_features = [x for x in df.columns if "skew" in x]
    for feature in skew_features:
        df[feature].fillna(0)
    df = df.dropna()
    return df


def read_and_clean_features(data_types, labels):
    df = pd.DataFrame()
    for data_type in data_types:
        features_data_path = get_data_path(data_type, "features")
        df_dt = pd.read_csv(f"{features_data_path}/all.csv")
        df_dt = df_dt.drop(["source", "sample"], axis=1)
        df = pd.concat([df, df_dt])
    df = clean_feature_data(df)

    if len(features) == 0:
        feature_df = df
    else:
        feature_df = df[features+labels]

    return feature_df


def get_model():
    estimator = MLPClassifier(hidden_layer_sizes=(500,250,100,50))
    clf = make_pipeline(StandardScaler(), estimator)
    return clf
