import os

import pandas as pd

from common import (cell_type_map, dimension, get_data_path)


def raw_to_processed():
    raw_data_path = get_data_path("in_silico", "raw")
    processed_data_path = get_data_path("in_silico", "processed")
    for experiment_dir in os.listdir(raw_data_path):
        experiment_path = f"{raw_data_path}/{experiment_dir}"
        if os.path.isfile(experiment_path):
            continue
        for rep_dir in os.listdir(experiment_path):
            rep_path = f"{experiment_path}/{rep_dir}"
            if os.path.isfile(rep_path):
                continue
            model_file = f"{rep_path}/{dimension}coords.csv"
            if not os.path.exists(model_file) or os.path.getsize(model_file) == 0:
                print(f"Data not found in {rep_path}")
                continue
            df = pd.read_csv(model_file)
            df = df[df["time"] == df["time"].max()]
            df = df[df["model"] == "nodrug"]
            df["type"] = df["type"].map(cell_type_map)
            df["sample"] = experiment_dir
            df = df[["sample", "type", "x", "y"]]
            df.to_csv(f"{processed_data_path}/spatial_{experiment_dir}.csv", index=False)


if __name__ == "__main__":
    raw_to_processed()