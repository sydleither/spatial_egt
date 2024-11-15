from collections import Counter
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from common import game_colors, get_data_path
from data_processing.processed_to_features import read_processed_sample
from data_processing.spatial_statistics import (create_nc_dists, 
                                                create_sfp_dist, 
                                                get_cell_type_counts)


def plot_dist(game_dists, save_loc, title, xlabel, ylabel, rnd):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for game in game_dists.keys():
        game_fs_counts = game_dists[game]
        game_fs_counts = [round(x, rnd) for x in game_fs_counts]
        color = game_colors[game]
        if "wins" in game:
            a = 0
        else:
            a = 1
        label = game
        counts = Counter(game_fs_counts)
        y_sum = sum(counts.values())
        freqs = [y/y_sum for y in counts.values()]
        ax[a].bar(counts.keys(), freqs, width=10**(-rnd), color=color, alpha=0.66, label=label)
    ax[0].legend()
    ax[1].legend()
    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    plt.savefig(f"{save_loc}/{title}.png", transparent=True)


def get_data(data_type, dist_func):
    cnt = 0
    game_dists = {"sensitive_wins":[], "coexistence":[],
                     "bistability":[], "resistant_wins":[]}
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    for file_name in os.listdir(processed_data_path):
        if file_name == "payoff.csv":
            continue
        df = read_processed_sample(processed_data_path,
                                   file_name, df_payoff)
        game = df["game"].iloc[0]
        if game == "unknown":
            continue
        s, r = get_cell_type_counts(df)
        prop_s = s/(s+r)
        if prop_s < 0.05 or prop_s > 0.95:
            continue
        s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
        r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)
        if dist_func == "sfp":
            dist = create_sfp_dist(s_coords, r_coords)
        elif dist_func == "nc":
            dist, _ = create_nc_dists(s_coords, r_coords, data_type)
        game_dists[game] += dist
        cnt += 1
        if cnt > 500:
            break
    return game_dists


def main(data_type):
    save_loc = get_data_path(data_type, "images")
    # all_fs_counts = get_data(data_type, "sfp")
    # plot_dist(all_fs_counts, save_loc, "Spatial Fokker-Planck Distributions",
    #           "Fraction Sensitive", "Frequency Across Subsamples", 1)
    all_fs = get_data(data_type, "nc")
    plot_dist(all_fs, save_loc, 
              "Neighborhood Composition Distributions",
              "Fraction Sensitive in Radius 3 from Resistant", 
              "Proportion of Resistant", 1)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the data type.")