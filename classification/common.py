import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (clean_feature_data, features,
                    game_colors, get_data_path)


def df_to_xy(df, label_name):
    feature_names = list(df.columns)
    feature_names.remove(label_name)
    label_categories = list(game_colors.keys())
    category_to_int = {lc:i for i,lc in enumerate(label_categories)}
    int_to_category = {i:lc for i,lc in enumerate(label_categories)}
    X = list(df[feature_names].values)
    y = [category_to_int[x] for x in df[label_name].values]
    return X, y, int_to_category


def read_and_clean_features(data_types):
    df = pd.DataFrame()
    for data_type in data_types:
        features_data_path = get_data_path(data_type, "features")
        df_dt = pd.read_csv(f"{features_data_path}/all.csv")
        df = pd.concat([df, df_dt])
    df = clean_feature_data(df)

    label = ["game"]
    if len(features) == 0:
        feature_df = df
    else:
        feature_df = df[features+label]
    X, y, int_to_name = df_to_xy(feature_df, label[0])

    return X, y, int_to_name


def get_model():
    estimator = MLPClassifier()#hidden_layer_sizes=(500,250,100,50))
    clf = make_pipeline(StandardScaler(), estimator)
    return clf