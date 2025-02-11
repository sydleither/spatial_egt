import os
import sys

import pandas as pd

from common import get_data_path, read_payoff_df


def main(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    features_data_path = get_data_path(data_type, "features")
    df = read_payoff_df(processed_data_path)[["game"]]
    for feature_set in os.listdir(features_data_path):
        if feature_set == "all.csv":
            continue
        df_feature_set = pd.read_csv(f"{features_data_path}/{feature_set}")
        df_feature_set["sample"] = df_feature_set["sample"].astype(str)
        df = pd.merge(df, df_feature_set, on=["source", "sample"])
    df.to_csv(f"{features_data_path}/all.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the data type.")