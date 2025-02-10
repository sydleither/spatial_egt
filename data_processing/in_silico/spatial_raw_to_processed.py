'''
Clean up coords of cells at final timestep and save in one spot.
extra argument is for other in silico experiments, such as 3D.
'''
import sys
import os

import pandas as pd

from common import cell_type_map, get_data_path


def main(extra=""):
    data_type = f"in_silico{extra}"
    raw_data_path = get_data_path(data_type, "raw")
    processed_data_path = get_data_path(data_type, "processed")
    for experiment_name in os.listdir(raw_data_path):
        experiment_path = f"{raw_data_path}/{experiment_name}"
        for data_dir in os.listdir(experiment_path):
            data_path = f"{experiment_path}/{data_dir}"
            if os.path.isfile(data_path):
                continue
            for rep_dir in os.listdir(data_path):
                rep_path = f"{data_path}/{rep_dir}"
                if os.path.isfile(rep_path):
                    continue
                for model_file in os.listdir(rep_path):
                    model_path = f"{rep_path}/{model_file}"
                    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                        print(f"Data not found in {model_path}")
                        continue
                    df = pd.read_csv(model_path)
                    df = df[df["time"] == df["time"].max()]
                    df = df[df["model"] == "nodrug"]
                    df["type"] = df["type"].map(cell_type_map)
                    cols_to_keep = ["type", "x", "y"]
                    if "z" in df:
                        cols_to_keep.append("z")
                    df = df[cols_to_keep]
                    df.to_csv(f"{processed_data_path}/{experiment_name} {data_dir}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(extra=sys.argv[1])
    else:
        main()
