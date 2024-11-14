import os
import sys

import pandas as pd

from common import get_data_path
from data_processing.spatial_statistics import calculate_game, process_sample


def processed_to_features(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    features_data_path = get_data_path(data_type, "features")
    df_entries = []
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    for file_name in os.listdir(processed_data_path):
        if file_name == "payoff.csv":
            continue
        source = file_name.split("_")[1]
        sample = file_name.split("_")[2][:-4]
        df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
        sample_payoff = df_payoff[(df_payoff["source"] == source) & (df_payoff["sample"] == sample)]
        features = process_sample(df_sample)
        features["game"] = calculate_game(sample_payoff)
        df_entries.append(features)
    df = pd.DataFrame(df_entries)
    df.to_csv(f"{features_data_path}/all.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        processed_to_features(sys.argv[1])
    else:
        print("Please provide the data type.")