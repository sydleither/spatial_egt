import os

import pandas as pd

from common import get_data_path, in_vitro_exp_names
from data_processing.spatial_statistics import calculate_game, process_sample


def processed_to_features():
    processed_data_path = get_data_path("in_vitro", "processed")
    features_data_path = get_data_path("in_vitro", "features")

    df_payoff = pd.DataFrame()
    for exp_name in in_vitro_exp_names:
        df_payoffs_plate = pd.read_csv(f"{processed_data_path}/payoff_{exp_name}.csv")
        df_payoff = pd.concat([df_payoff, df_payoffs_plate])
    
    df_entries = []
    for sample in os.listdir(processed_data_path):
        if sample.startswith("payoff"):
            continue
        plate = sample.split("_")[1]+"_plate1"
        well = sample.split("_")[2][:-4]
        df_sample = pd.read_csv(f"{processed_data_path}/{sample}")
        sample_payoff = df_payoff[(df_payoff["PlateId"] == plate) & (df_payoff["WellId"] == well)]
        features = process_sample(df_sample)
        features["game"] = calculate_game(sample_payoff)
        df_entries.append(features)
    df = pd.DataFrame(df_entries)
    df.to_csv(f"{features_data_path}/features.csv", index=False)


if __name__ == "__main__":
    processed_to_features()