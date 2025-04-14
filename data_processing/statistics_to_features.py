import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from spatial_egt.common import get_data_path, read_payoff_df


def distribution_to_features(row, name):
    dist = row[name]
    row[f"{name}_Mean"] = np.mean(dist)
    sd = np.std(dist)
    row[f"{name}_SD"] = sd
    if sd == 0:
        row[f"{name}_Skew"] = 0
        row[f"{name}_Kurtosis"] = 0
    else:
        row[f"{name}_Skew"] = skew(dist)
        row[f"{name}_Kurtosis"] = kurtosis(dist, fisher=True, bias=True)
    return row


def function_to_features(row, name):
    func = row[name]
    row[f"{name}_Min"] = min(func)
    row[f"{name}_Max"] = max(func)
    return row


def main(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    statistics_data_path = get_data_path(data_type, "statistics")
    df = read_payoff_df(processed_data_path)[["game"]]
    for statistic_file in os.listdir(statistics_data_path):
        if not statistic_file.endswith(".pkl"):
            continue
        df_feature = pd.read_pickle(f"{statistics_data_path}/{statistic_file}")
        statistic_name = statistic_file[:-4]
        statistic_type = df_feature["type"].iloc[0]
        if statistic_type == "distribution":
            df_feature = df_feature.apply(distribution_to_features, axis=1, args=(statistic_name,))
            df_feature = df_feature.drop(statistic_name, axis=1)
        elif statistic_type == "function":
            df_feature = df_feature.apply(function_to_features, axis=1, args=(statistic_name,))
            df_feature = df_feature.drop(statistic_name, axis=1)
        df_feature = df_feature.drop("type", axis=1)
        df_feature["sample"] = df_feature["sample"].astype(str)
        df = pd.merge(df, df_feature, on=["source", "sample"], how="outer")
    df.to_csv(f"{statistics_data_path}/features.csv", index=False, na_rep=np.nan)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the data type.")
