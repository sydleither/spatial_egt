'''
Compile configs into a csv with the payoff values and game of each sample/model.
extra argument is for other in silico experiments, such as 3D.
'''
import json
import os
import sys

import pandas as pd

from common import calculate_game, get_data_path


def main(extra=""):
    data_type = f"in_silico_{extra}"
    raw_data_path = get_data_path(data_type, "raw")
    processed_data_path = get_data_path(data_type, "processed")
    df_entries = []
    for experiment_name in os.listdir(raw_data_path):
        experiment_path = f"{raw_data_path}/{experiment_name}"
        for data_dir in os.listdir(experiment_path):
            data_path = f"{experiment_path}/{data_dir}"
            if os.path.isfile(data_path):
                continue
            df_row = {}
            config = json.load(open(f"{data_path}/{data_dir}.json"))
            df_row["source"] = experiment_name
            df_row["sample"] = data_dir
            df_row["initial_density"] = config["numCells"]
            df_row["initial_fr"] = config["proportionResistant"]
            df_row["a"] = config["A"]
            df_row["b"] = config["B"]
            df_row["c"] = config["C"]
            df_row["d"] = config["D"]
            df_row["game"] = calculate_game(config["A"], config["B"], config["C"], config["D"])
            df_entries.append(df_row)
    df = pd.DataFrame(data=df_entries)
    df.to_csv(f"{processed_data_path}/payoff.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(extra=sys.argv[1])
    else:
        main()
