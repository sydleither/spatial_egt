import os
import sys

import pandas as pd

from common import get_data_path, read_payoff_df
from data_processing.spatial_statistics import sample_to_features


def processed_to_features(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    features_data_path = get_data_path(data_type, "features")
    df_entries = []
    df_payoff = read_payoff_df(processed_data_path)
    for file_name in os.listdir(processed_data_path):
        if file_name == "payoff.csv":
            continue
        df_sample = pd.read_csv(f"{processed_data_path}/payoff.csv")
        features = sample_to_features(df_sample, data_type)
        sample = file_name.split("_")[2][:-4]
        source = file_name.split("_")[1]
        features["game"] = df_payoff.at[(sample, source), "game"]
        df_entries.append(features)
    df = pd.DataFrame(df_entries)
    df.to_csv(f"{features_data_path}/all.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        processed_to_features(sys.argv[1])
    else:
        print("Please provide the data type.")