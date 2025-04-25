"""Convert spatial statistics into features and save as features.csv

Features are defined as the single-valued statistics and the summary
    statistics of the distribution and function statistics.

Features.csv also contains a column with the "class label" for downstream
    machine learning or analysis.

Expected usage:
python3 -m spatial_egt.data_processing.statistics_to_features
    data_type label_name

Where:
data_type: the name of the directory in data/
label_name: the class label name, which also exists in data/{data_type}/labels.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew

from spatial_egt.common import get_data_path, get_spatial_statistic_type


def distribution_to_features(row, name):
    """Distribution summary statistics"""
    dist = row[name]
    row[f"{name}_Mean"] = np.mean(dist)
    sd = np.std(dist)
    row[f"{name}_SD"] = sd
    if sd == 0:
        row[f"{name}_Skew"] = 0
    else:
        row[f"{name}_Skew"] = skew(dist)
    return row


def function_to_features(row, name):
    """Function summary statistics"""
    func = row[name]
    row[f"{name}_Min"] = min(func)
    row[f"{name}_Max"] = max(func)
    return row


def main(data_type, label_name):
    """Convert spatial statistics into features"""
    data_path = get_data_path(data_type, ".")
    statistics_data_path = get_data_path(data_type, "statistics")
    df = pd.read_csv(f"{data_path}/labels.csv")
    df["sample"] = df["sample"].astype(str)
    df = df[["source", "sample", label_name]]
    for statistic_file in os.listdir(statistics_data_path):
        if not statistic_file.endswith(".pkl"):
            continue
        df_feature = pd.read_pickle(f"{statistics_data_path}/{statistic_file}")
        statistic_name = statistic_file[:-4]
        statistic_type = get_spatial_statistic_type(df_feature, statistic_name)
        if statistic_type == "distribution":
            df_feature = df_feature.apply(distribution_to_features, axis=1, args=(statistic_name,))
            df_feature = df_feature.drop(statistic_name, axis=1)
        elif statistic_type == "function":
            df_feature = df_feature.apply(function_to_features, axis=1, args=(statistic_name,))
            df_feature = df_feature.drop(statistic_name, axis=1)
        df_feature["sample"] = df_feature["sample"].astype(str)
        df = pd.merge(df, df_feature, on=["source", "sample"], how="outer")
    df.to_csv(f"{statistics_data_path}/features.csv", index=False, na_rep=np.nan)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        print("Please see the module docstring for usage instructions.")
