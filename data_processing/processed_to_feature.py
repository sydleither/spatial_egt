import os
import sys

from pandas.api.types import is_object_dtype
import pandas as pd

from common import get_data_path
from data_processing.write_feature_jobs import FEATURE_PARAMS, FEATURE_REGISTRY


def main(feature_name, data_type):
    processed_path = get_data_path(data_type, "processed")
    features_path = get_data_path(data_type, "features")

    feature_calculation = FEATURE_REGISTRY[feature_name]
    feature_args_datatype = FEATURE_PARAMS[data_type]
    feature_args = dict()
    if feature_name in feature_args_datatype:
        feature_args = feature_args_datatype[feature_name]

    df_entries = []
    for file_name in os.listdir(processed_path):
        if file_name == "payoff.csv":
            continue
        df_sample = pd.read_csv(f"{processed_path}/{file_name}")
        try:
            feature = feature_calculation(df_sample, **feature_args)
        except Exception as e:
            print(f"Error: {file_name}")
            print(e)
            continue
        source = file_name.split(" ")[0]
        sample = file_name.split(" ")[1][:-4]
        df_entries.append({"source":source, "sample":sample, feature_name:feature})

    df = pd.DataFrame(df_entries)
    if is_object_dtype(df[feature_name]):
        if feature_calculation.__name__.endswith("dist"):
            df["type"] = "distribution"
        else:
            df["type"] = "function"
    else:
        df["type"] = "value"
    df.to_pickle(f"{features_path}/{feature_name}.pkl")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the feature name and data type.")
