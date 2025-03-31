import os
import sys

from pandas.api.types import is_object_dtype
import pandas as pd

from common import get_data_path
from data_processing.write_feature_jobs import FEATURE_PARAMS, FEATURE_REGISTRY


def main(data_type, feature_name, source, sample):
    processed_path = get_data_path(data_type, "processed")
    features_path = get_data_path(data_type, f"features/{feature_name}")

    feature_calculation = FEATURE_REGISTRY[feature_name]
    feature_args_datatype = FEATURE_PARAMS[data_type]
    feature_args = dict()
    if feature_name in feature_args_datatype:
        feature_args = feature_args_datatype[feature_name]

    df_entries = []
    df_sample = pd.read_csv(f"{processed_path}/{source} {sample}.csv")
    feature = feature_calculation(df_sample, **feature_args)
    df_entries.append({"source":source, "sample":sample, feature_name:feature})

    df = pd.DataFrame(df_entries)
    if is_object_dtype(df[feature_name]):
        if feature_calculation.__name__.endswith("dist"):
            df["type"] = "distribution"
        else:
            df["type"] = "function"
    else:
        df["type"] = "value"
    df.to_pickle(f"{features_path}/{sample}.pkl")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Please provide the data type, feature name, source, and sample.")
