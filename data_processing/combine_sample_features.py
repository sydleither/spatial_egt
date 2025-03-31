import os
import sys

import pandas as pd

from common import get_data_path


def main(data_type, feature_name):
    features_path = get_data_path(data_type, f"features/{feature_name}")
    df = pd.DataFrame()
    for file_name in os.listdir(features_path):
        df_sample = pd.read_pickle(f"{features_path}/{file_name}")
        df = pd.merge([df, df_sample])
    df.to_pickle(f"{features_path}/{feature_name}.pkl")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the data type and feature name.")
