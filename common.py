import pandas as pd

import os


dimension = "2D"
in_vitro_exp_names = ["braf", "nls"]
cell_type_map = {0: "sensitive", 1:"resistant",
                 "S-3E9": "sensitive", "BRAF-mCherry":"resistant",
                 "S-NLS": "sensitive", "R-NLS": "resistant", "mCherry": "resistant",
                 "Red":"sensitive", "Green":"resistant", "Blue":"unknown"}
game_colors = {"sensitive_wins":"#4C956C", "coexistence":"#F97306",
               "bistability":"#047495", "resistant_wins":"#EF7C8E"}


def get_data_path(data_type, data_stage):
    data_path = f"data/{data_type}/{data_stage}"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path


def read_payoff_df(processed_data_path):
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    df_payoff = df_payoff.set_index(["sample", "source"], drop=False)
    return df_payoff


def calculate_game(a, b, c, d):
    if a > c and b > d:
        game = "sensitive_wins"
    elif c > a and b > d:
        game = "coexistence"
    elif a > c and d > b:
        game = "bistability"
    elif c > a and d > b:
        game = "resistant_wins"
    else:
        game = "unknown"
    return game
