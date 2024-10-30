import os

import pandas as pd

from common import get_data_path
from data_processing.spatial_statistics import calculate_game, process_sample


def processed_to_features():
    processed_data_path = get_data_path("in_silico", "processed")
    features_data_path = get_data_path("in_silico", "features")
    df_entries = []
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    for sample in os.listdir(processed_data_path):
        if sample == "payoff.csv":
            continue
        sample_id = int(sample.split("_")[1][:-4])
        df_sample = pd.read_csv(f"{processed_data_path}/{sample}")
        sample_payoff = df_payoff[df_payoff["sample"] == sample_id]
        features = process_sample(df_sample)
        features["game"] = calculate_game(sample_payoff)
        df_entries.append(features)
    df = pd.DataFrame(df_entries)
    df.to_csv(f"{features_data_path}/all.csv", index=False)


if __name__ == "__main__":
    processed_to_features()