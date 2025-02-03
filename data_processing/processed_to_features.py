import os
import sys

import pandas as pd

from common import get_data_path, read_payoff_df
from data_processing.spatial_statistics import sample_to_features


def main(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    features_data_path = get_data_path(data_type, "features")
    df_entries = []
    df_payoff = read_payoff_df(processed_data_path)
    for file_name in os.listdir(processed_data_path):
        if file_name == "payoff.csv":
            continue
        df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
        features = sample_to_features(df_sample, data_type)
        source = file_name.split(" ")[0]
        sample = file_name.split(" ")[1][:-4]
        features["game"] = df_payoff.at[(source, sample), "game"]
        df_entries.append(features)
    df = pd.DataFrame(df_entries)
    df.to_csv(f"{features_data_path}/all.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the data type.")