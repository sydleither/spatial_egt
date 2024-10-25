import json
import os

import pandas as pd

from common import get_data_path


def raw_to_processed():
    raw_data_path = get_data_path("in_silico", "raw")
    processed_data_path = get_data_path("in_silico", "processed")
    df_entries = []
    for experiment_dir in os.listdir(raw_data_path):
        experiment_path = f"{raw_data_path}/{experiment_dir}"
        if os.path.isfile(experiment_path):
            continue
        df_row = {}
        config = json.load(open(f"{experiment_path}/{experiment_dir}.json"))
        df_row["sample"] = experiment_dir
        df_row["initial_density"] = config["numCells"]
        df_row["initial_fr"] = config["proportionResistant"]
        df_row["a"] = config["A"]
        df_row["b"] = config["B"]
        df_row["c"] = config["C"]
        df_row["d"] = config["D"]
        df_entries.append(df_row)
    df = pd.DataFrame(data=df_entries)
    df.to_csv(f"{processed_data_path}/payoff.csv", index=False)


if __name__ == "__main__":
    raw_to_processed()