import os
import sys

import pandas as pd

from common import get_data_path, read_payoff_df
from data_processing.spatial_statistics import (create_custom_features,
                                                create_muspan_features)


def main(feature_set, data_type):
    # Set which feature set to calculate
    if feature_set == "custom":
        sample_to_features = create_custom_features
    elif feature_set == "muspan":
        sample_to_features = create_muspan_features
    else:
        print(f"Invalid feature set provided: {feature_set}")
        return

    # Set variables
    processed_data_path = get_data_path(data_type, "processed")
    features_data_path = get_data_path(data_type, "features")
    df_payoff = read_payoff_df(processed_data_path)
    df_entries = []

    # Figure out dimension of data based on sample file
    file_names = os.listdir(processed_data_path)
    sample_df = pd.read_csv(f"{processed_data_path}/{file_names[0]}")
    dimensions = list(sample_df.drop("type", axis=1).columns)

    # Calculate features for each sample
    for file_name in file_names:
        if file_name == "payoff.csv":
            continue
        df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
        features = sample_to_features(df_sample, data_type, dimensions)
        source = file_name.split(" ")[0]
        sample = file_name.split(" ")[1][:-4]
        features["game"] = df_payoff.at[(source, sample), "game"]
        df_entries.append(features)

    # Save calculated features
    df = pd.DataFrame(df_entries)
    df.to_csv(f"{features_data_path}/{feature_set}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the feature set type (custom, muspan) and data type.")
