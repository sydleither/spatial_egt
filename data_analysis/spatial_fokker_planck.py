from collections import Counter
import os

import matplotlib.pyplot as plt
import pandas as pd

from common import cell_colors, get_data_path
from data_processing.spatial_statistics import calculate_game, get_cell_type_counts


def plot_fps(all_fs_counts, save_loc):
    colors = {"sensitive_wins":cell_colors[0], "coexistence":"#f97306",
              "bistability":"#029386", "resistant_wins":cell_colors[1]}
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for game in all_fs_counts.keys():
        game_fs_counts = all_fs_counts[game]
        color = colors[game]
        if "wins" in game:
            a = 0
        else:
            a = 1
        label = game
        counts = Counter(game_fs_counts)
        y_sum = sum(counts.values())
        freqs = [y/y_sum for y in counts.values()]
        ax[a].bar(counts.keys(), freqs, width=0.1, color=color, alpha=0.75, label=label)
    ax[0].legend()
    ax[1].legend()
    fig.suptitle("Spatial Fokker-Planck Distributions\nAcross 1000 Random Samples\nwith Differing Initital Density, Initital Fraction Sensitive, and Payoff")
    fig.supxlabel("Fraction Sensitive")
    fig.supylabel("Frequency Across Subsamples")
    fig.tight_layout()
    plt.savefig(f"{save_loc}/spatial_fokker_planck.png", transparent=True)


def calculate_dist(s_coords, r_coords):
    all_coords = [("s", s_coords[i][0], s_coords[i][1]) for i in range(len(s_coords))]
    all_coords += [("r", r_coords[i][0], r_coords[i][1]) for i in range(len(r_coords))]

    max_x = max([x[1] for x in all_coords])
    max_y = max([x[2] for x in all_coords])
    max_coord = max(max_x, max_y)

    fs_counts = []
    subset_size = 10
    for s in range(max_coord//subset_size):
        lower = s*subset_size
        upper = (s+1)*subset_size
        subset = [t for t,x,y in all_coords if lower <= x <= upper and lower <= y <= upper]
        subset_total = len(subset)
        subset_s = len([x for x in subset if x[0] == "s"])
        if subset_total < 10:
            continue
        fs_counts.append(round(subset_s/subset_total, 1))
    return fs_counts


def get_data():
    cnt = 0
    all_fs_counts = {"sensitive_wins":[], "coexistence":[],
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
        fs_counts = calculate_dist(s_coords, r_coords)
        all_fs_counts[game] += fs_counts
        cnt += 1
        if cnt > 1000:
            break
    return all_fs_counts


def main():
    all_fs_counts = get_data()
    save_loc = get_data_path("in_silico", "images")
    plot_fps(all_fs_counts, save_loc)


if __name__ == "__main__":
    main()
        