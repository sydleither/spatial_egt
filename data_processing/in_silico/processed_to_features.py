import os
import sys

import pandas as pd

from common import get_data_path
from data_processing.spatial_statistics import process_sample


def processed_to_features():
    processed_data_path = get_data_path("in_silico", "processed")
    features_data_path = get_data_path("in_silico", "features")

    df_entries = []
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    for sample in os.listdir(processed_data_path):
        if sample == "payoff.csv":
            continue
        df_sample = pd.read_csv(f"{processed_data_path}/{sample}")
        df_feature = process_sample(df_sample)
        exit()


if __name__ == "__main__":
    processed_to_features()