import os
import sys

import pandas as pd

from common import get_data_path
from data_processing.spatial_statistics import calculate_game, sample_to_features


def read_processed_sample(processed_data_path, file_name, df_payoff):
    source = file_name.split("_")[1]
    sample = file_name.split("_")[2][:-4]
    df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
    df_sample["source"] = source
    df_sample["sample"] = sample
    sample_payoff = df_payoff[(df_payoff["source"] == source) & (df_payoff["sample"] == sample)]
    game = sample_payoff.apply(calculate_game, axis="columns").iloc[0]
    df_sample["game"] = game
    return df_sample


def processed_to_features(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    features_data_path = get_data_path(data_type, "features")
    df_entries = []
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    for file_name in os.listdir(processed_data_path):
        if file_name == "payoff.csv":
            continue
        df_sample = read_processed_sample(processed_data_path, 
                                          file_name, df_payoff)
        features = sample_to_features(df_sample, data_type)
        features["game"] = df_sample["game"].iloc[0]
        df_entries.append(features)
    df = pd.DataFrame(df_entries)
    df.to_csv(f"{features_data_path}/all.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        processed_to_features(sys.argv[1])
    else:
        print("Please provide the data type.")