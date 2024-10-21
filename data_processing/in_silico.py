import json
import os
import sys

import pandas as pd

from spatial_statistics import create_all_features


def process_df(df):
    df = df.loc[(df["time"] == df["time"].max())]
    return df


def save_data(exp_dir, dimension):
    df_entries = []
    exp_path = f"data/in_silico/raw/{exp_dir}"
    uid = 0
    extinct = 0
    unknwon_game = 0
    for game_dir in os.listdir(exp_path):
        game_path = f"{exp_path}/{game_dir}"
        if os.path.isfile(game_path):
            continue
        config = json.load(open(f"{game_path}/{game_dir}.json"))
        a = config["A"]
        b = config["B"]
        c = config["C"]
        d = config["D"]
        cells = config["numCells"]
        fr = config["proportionResistant"]
        for rep_dir in os.listdir(game_path):
            rep_path = f"{game_path}/{rep_dir}"
            if os.path.isfile(rep_path):
                continue
            model_file = f"{rep_path}/{dimension}coords.csv"
            if not os.path.exists(model_file) or os.path.getsize(model_file) == 0:
                print(f"File not found in {rep_path}")
                continue
            sample_dict = {}
            sample_dict["rep"] = int(rep_dir)
            sample_dict["initial_fr"] = fr
            sample_dict["initial_cells"] = cells
            sample_dict["A"] = a
            sample_dict["B"] = b
            sample_dict["C"] = c
            sample_dict["D"] = d
            if a > c and b > d:
                game = "sensitive_wins"
            elif a < c and b > d:
                game = "coexistence"
            elif a > c and b < d:
                game = "bistability"
            elif a < c and b < d:
                game = "resistant_wins"
            else:
                game = "unknown"
            sample_dict["game"] = game
            uid += 1
            sample_dict["uid"] = uid
            df_coords = process_df(pd.read_csv(model_file))
            num_sensitive = len(df_coords.loc[df_coords["type"] == 0])
            num_resistant = len(df_coords.loc[df_coords["type"] == 1])
            if num_resistant < 100 or num_sensitive < 100:
                extinct += 1
                continue
            if game == "unknown":
                unknwon_game += 1
                continue
            feature_dict = create_all_features(df_coords, num_sensitive, num_resistant)
            sample_dict = sample_dict | feature_dict
            df_entries.append(sample_dict)
        if uid % 100 == 0:
            print(f"Processed {uid} samples...")
    print(f"Skipped {extinct} samples nearing extinction.")
    print(f"Skipped {unknwon_game} samples with unknown games.")
    print(f"Total samples: {len(df_entries)}")
    df = pd.DataFrame(data=df_entries)
    pd.to_pickle(df, f"data/in_silico/processed/{exp_dir}/{dimension}df.pkl")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        save_data(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the experiment directory and dimension.")