from collections import Counter
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import KDTree

from common import game_colors, get_data_path
from data_processing.spatial_statistics import calculate_game, get_cell_type_counts


def plot_dist(fi, save_loc, main_cell, neighbor_cell):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for game in fi.keys():
        game_fs_counts = fi[game]
        color = game_colors[game]
        if "wins" in game:
            a = 0
        else:
            a = 1
        label = game
        counts = Counter(game_fs_counts)
        print(counts)
        y_sum = sum(counts.values())
        freqs = [y/y_sum for y in counts.values()]
        ax[a].bar(counts.keys(), freqs, width=0.01, color=color, alpha=0.75, label=label)
        ax[a].set(ylim=(0,0.05))
    ax[0].legend()
    ax[1].legend()
    fig.suptitle("Neighborhood Composition Distributions\nAcross 1000 Random Samples\nwith Differing Initital Density, Initital Fraction Sensitive, and Payoff")
    fig.supxlabel(f"Fraction {neighbor_cell} in Radius 3 from {main_cell}")
    fig.supylabel(f"Proportion of {main_cell}")
    fig.tight_layout()
    plt.savefig(f"{save_loc}/neighborhood_composition_{main_cell}.png", transparent=True)


def calculate_dist(s_coords, r_coords):
    all_coords = s_coords + r_coords
    radius = 3

    s_stop = len(s_coords)
    tree = KDTree(all_coords)
    fs = []
    fr = []
    for p,point in enumerate(all_coords):
        neighbor_indices = tree.query_ball_point(point, radius)
        all_neighbors = len(neighbor_indices)-1
        if p <= s_stop: #sensitive cell
            r_neighbors = len([x for x in neighbor_indices if x > s_stop])
            if all_neighbors != 0 and r_neighbors != 0:
                fr.append(round(r_neighbors/all_neighbors, 2))
        else: #resistant cell
            s_neighbors = len([x for x in neighbor_indices if x <= s_stop])
            if all_neighbors != 0 and s_neighbors != 0:
                fs.append(round(s_neighbors/all_neighbors, 1))

    return fr, fs


def get_data():
    cnt = 0
    all_fs = {"sensitive_wins":[], "coexistence":[],
                     "bistability":[], "resistant_wins":[]}
    all_fr = {"sensitive_wins":[], "coexistence":[],
                     "bistability":[], "resistant_wins":[]}
    processed_data_path = get_data_path("in_silico", "processed")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    for sample in os.listdir(processed_data_path):
        if sample == "payoff.csv":
            continue
        sample_id = int(sample.split("_")[1][:-4])
        df = pd.read_csv(f"{processed_data_path}/{sample}")
        payoff = df_payoff[df_payoff["sample"] == sample_id]
        game = calculate_game(payoff)
        if game == "unknown":
            continue
        s, r = get_cell_type_counts(df)
        if s < 100 or r < 100:
            continue
        s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
        r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)
        fr, fs = calculate_dist(s_coords, r_coords)
        all_fs[game] += fs
        all_fr[game] += fr
        cnt += 1
        if cnt > 1000:
            break
    return all_fr, all_fs


def main():
    all_fr, all_fs = get_data()
    save_loc = get_data_path("in_silico", "images")
    plot_dist(all_fr, save_loc, "Sensitive", "Resistant")
    plot_dist(all_fs, save_loc, "Resistant", "Sensitive")


if __name__ == "__main__":
    main()
        