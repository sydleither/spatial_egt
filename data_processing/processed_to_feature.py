import os
import sys

from pandas.api.types import is_object_dtype
import pandas as pd

from common import get_data_path
from data_processing.write_feature_jobs import FEATURE_PARAMS, FEATURE_REGISTRY


def calculate_features(processed_path, file_names, feature_name, feature_calculation, feature_args):
    rows = []
    for file_name in file_names:
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
        rows.append({"source":source, "sample":sample, feature_name:feature})
    return rows


def main(data_type, feature_name, source=None, sample=None):
    processed_path = get_data_path(data_type, "processed")

    feature_calculation = FEATURE_REGISTRY[feature_name]
    feature_args_datatype = FEATURE_PARAMS[data_type]
    feature_args = dict()
    if feature_name in feature_args_datatype:
        feature_args = feature_args_datatype[feature_name]

    if source is None and sample is None:
        features_path = get_data_path(data_type, f"features")
        save_loc = f"{features_path}/{feature_name}.pkl"
        file_names = os.listdir(processed_path)
    else:
        features_path = get_data_path(data_type, f"features/{feature_name}")
        save_loc = f"{features_path}/{sample}.pkl"
        file_names = [f"{source} {sample}.csv"]
    
    rows = calculate_features(processed_path, file_names, feature_name, feature_calculation, feature_args)

    df = pd.DataFrame(rows)
    if is_object_dtype(df[feature_name]):
        if feature_calculation.__name__.endswith("dist"):
            df["type"] = "distribution"
        else:
            df["type"] = "function"
    else:
        df["type"] = "value"
    df.to_pickle(save_loc)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the data type and feature name")
        print("and a source and sample id, if calculating individual samples.")
