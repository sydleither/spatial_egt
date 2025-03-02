import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew

from common import get_data_path, read_payoff_df


def get_dist_statistics(row, name):
    dist = row[name]
    row[f"{name} Mean"] = np.mean(dist)
    row[f"{name} SD"] = np.std(dist)
    row[f"{name} Skew"] = skew(dist)
    return row


def get_func_statistics(row, name):
    func = row[name]
    row[f"{name} 0"] = func[0]
    row[f"{name} Min"] = min(func)
    row[f"{name} Max"] = max(func)
    return row


def main(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    features_data_path = get_data_path(data_type, "features")
    df = read_payoff_df(processed_data_path)[["game"]]
    for feature_file in os.listdir(features_data_path):
        if feature_file == "all.csv":
            continue
        df_feature = pd.read_pickle(f"{features_data_path}/{feature_file}")
        feature_name = feature_file[:-4]
        feature_type = df_feature["type"].iloc[0]
        if feature_type == "distribution":
            df_feature = df_feature.apply(get_dist_statistics, axis=1, args=(feature_name,))
            df_feature = df_feature.drop(feature_name, axis=1)
        elif feature_type == "function":
            df_feature = df_feature.apply(get_func_statistics, axis=1, args=(feature_name,))
            df_feature = df_feature.drop(feature_name, axis=1)
        df_feature = df_feature.drop("type", axis=1)
        df_feature["sample"] = df_feature["sample"].astype(str)
        df = pd.merge(df, df_feature, on=["source", "sample"])
    df.to_csv(f"{features_data_path}/all.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the data type.")