import pandas as pd

import os


game_colors = {"Sensitive Wins":"#4C956C", "Coexistence":"#F97306",
               "Bistability":"#047495", "Resistant Wins":"#EF7C8E"}
theme_colors = ["xkcd:faded purple", "xkcd:lemon yellow"]


def get_data_path(data_type, data_stage):
    data_path = f"data/{data_type}/{data_stage}"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path


def read_payoff_df(processed_data_path):
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    df_payoff = df_payoff.set_index(["source", "sample"], drop=False)
    return df_payoff


def calculate_game(a, b, c, d):
    if a > c and b > d:
        game = "Sensitive Wins"
    elif c > a and b > d:
        game = "Coexistence"
    elif a > c and d > b:
        game = "Bistability"
    elif c > a and d > b:
        game = "Resistant Wins"
    else:
        game = "Unknown"
    return game
