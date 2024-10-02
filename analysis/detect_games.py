from itertools import chain, combinations
import os
import sys

import pandas as pd

from common import process_fs
sys.path.insert(0, "DDIT")
from DDIT import DDIT


def save_data(exp_dir, dimension):
    df = pd.DataFrame()
    exp_path = f"output/{exp_dir}"
    uid = 0
    for grid_params in os.listdir(exp_path):
        grid_path = f"{exp_path}/{grid_params}"
        if os.path.isfile(grid_path):
            continue
        fr = int(grid_params[2])/10
        cells = int(grid_params.split("_")[-1][1:])
        for game_dir in os.listdir(grid_path):
            game_path = f"{grid_path}/{game_dir}"
            if os.path.isfile(game_path):
                continue
            game = game_dir.split("_")[0]
            subgame = game_dir.split("_")[1]
            for rep_dir in os.listdir(game_path):
                rep_path = f"{game_path}/{rep_dir}"
                if os.path.isfile(rep_path):
                    continue
                pop_file = f"{rep_path}/{dimension}populations.csv"
                fs_file = f"{rep_path}/{dimension}fs.csv"
                if not os.path.exists(pop_file) or os.path.getsize(pop_file) == 0 or\
                   not os.path.exists(fs_file) or os.path.getsize(fs_file) == 0:
                    print(f"File not found in {rep_path}")
                    continue
                df_pop_i = pd.read_csv(pop_file)
                df_pop_i["dimension"] = dimension
                df_pop_i["rep"] = int(rep_dir)
                df_pop_i["initial_fr"] = fr
                df_pop_i["initial_cells"] = cells
                df_pop_i["game"] = game
                df_pop_i["subgame"] = subgame
                df_pop_i["uid"] = uid
                df_fs_i = pd.read_csv(fs_file)
                df_fs_i = process_fs(df_fs_i, ["model", "time", "radius"])
                df_i = df_fs_i.merge(df_pop_i, on=["model", "time"])
                df = pd.concat([df, df_i])
                uid += 1
    pd.to_pickle(df, f"output/{exp_dir}/{dimension}df.pkl")


def create_fs_features(df):
    #slope of average fs over neighborhood radii
    df_r1 = df.loc[df["radius"] == 1]
    df_r1["avg_fs_r1"] = df_r1["average_fs"]
    df_r5 = df.loc[df["radius"] == 5]
    df_r5["avg_fs_r5"] = df_r5["average_fs"]
    df_slope = df_r1[["uid", "avg_fs_r1"]].merge(df_r5[["uid", "avg_fs_r5"]], on=["uid"])
    df_slope = df_slope.drop_duplicates()
    df_slope["avg_fs_slope"] = df_slope["avg_fs_r5"] - df_slope["avg_fs_r1"]
    df_slope = df_slope.drop(columns=["avg_fs_r1", "avg_fs_r5"])
    df = df.merge(df_slope, on=["uid"])

    #proportion of boundary cells
    df_r1 = df.loc[df["radius"] == 1]
    df_r1["s_in_neighborhood"] = df_r1["total_boundary"] / df_r1["resistant"]
    df_r1 = df_r1[["uid", "s_in_neighborhood"]].drop_duplicates()
    df = df.merge(df_r1, on=["uid"])

    #average fs
    df_r3 = df.loc[df["radius"] == 3]
    df = df.drop(columns=["average_fs"])
    df_r3 = df_r3[["uid", "average_fs"]].drop_duplicates()
    df = df.merge(df_r3, on=["uid"])

    #collaspe
    df = df.drop(columns=["radius", "fs", "total", "reproduced", "weighted_fs", 
                          "total_boundary", "weighted_fs_sum", "normalized_total"])
    df = df.drop_duplicates()
    
    return df


def create_pop_features(df):
    #proportion of cells that are resistant
    df["proportion_resistant"] = df["resistant"] / (df["resistant"] + df["sensitive"])

    #amount of grid filled
    df["density"] = (df["resistant"] + df["sensitive"]) / 15625

    return df


def create_fragmentation_matrix(df):
    ddit = DDIT()
    for column in df.columns:
        column_data = df[column].values
        if column != "game":
            column_data = [round(x,2) for x in column_data]
        column_data = tuple(column_data)
        ddit.register_column_tuple(column, column_data)
    features = list(df.columns)
    features.remove("game")
    feature_powerset = chain.from_iterable(combinations(features, r) for r in range(len(features)+1))
    game_entropy = ddit.H("game")
    for entry in feature_powerset:
        if len(entry) == 0:
            continue
        ent = ddit.recursively_solve_formula("game:"+"&".join(entry)) / game_entropy
        print("game:"+"&".join(entry), ent)


def main(exp_dir, dimension):
    try:
        df = pd.read_pickle(f"output/{exp_dir}/{dimension}df.pkl")
    except:
        print("Please save the dataframe.")
        exit()
    df = create_pop_features(create_fs_features(df))
    features = df.drop(columns=["model", "time", "sensitive", "resistant", "dimension",
                                "rep", "initial_fr", "initial_cells", "subgame", "uid"])
    create_fragmentation_matrix(features)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        if sys.argv[3] == "save":
            save_data(sys.argv[1], sys.argv[2])
        else:
            print("Please provide am experiment directory, dimension, and \"save\"")
            print("if the dataframe has not yet been saved.")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the experiment directory and dimension, if the dataframe has been saved.")