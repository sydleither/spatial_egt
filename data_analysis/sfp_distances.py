import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

from common import game_colors, get_data_path
from data_processing.processed_to_features import read_processed_sample
from data_processing.spatial_statistics import (calculate_game, create_sfp_dist)


def get_sfp_dist(processed_data_path, df_payoff, data_type, source, sample_id, subset_size, incl_empty):
    file_name = f"spatial_{source}_{sample_id}.csv"
    df = read_processed_sample(processed_data_path, file_name, df_payoff)
    s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
    r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)
    dist = create_sfp_dist(s_coords, r_coords, data_type, subset_size, 1000, incl_empty)
    return dist


def get_payoff_data(processed_data_path, data_type):
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    df_payoff["game"] = df_payoff.apply(calculate_game, axis="columns")
    df_payoff = df_payoff[df_payoff["game"] != "unknown"]
    return df_payoff


def get_theoretical_dists(n):
    rw_dist = stats.beta.rvs(a=1, b=5, loc=0.1, scale=0.75, size=n)
    co_dist = stats.norm.rvs(loc=0.5, scale=0.2, size=n)
    sw_dist = stats.beta.rvs(a=5, b=1, loc=0.25, scale=0.65, size=n)
    bi_dist = np.concatenate((np.random.choice(rw_dist, n//2), np.random.choice(sw_dist, n//2)), axis=None)
    dists = [sw_dist, co_dist, bi_dist, rw_dist]
    games = ["sensitive_wins", "coexistence", "bistability", "resistant_wins"]
    return dists, games


def plot_dists(save_loc, dists, games, n):
    freq = np.ones_like(range(n))/n
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(len(dists)):
        ax[i].hist(dists[i], bins=20, weights=freq, color=game_colors[games[i]])
        ax[i].set(xlim=(0, 1), ylim=(0, 0.2))
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/dists.png")
    plt.close()


def distances_plot(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = get_payoff_data(processed_data_path, data_type)
    samples = list(df_payoff[["sample", "source", "game"]].values)

    dists, games = get_theoretical_dists(1000)
    dists = {games[i]:dists[i] for i in range(len(games))}

    rows = []
    for sample_id, source, game in samples:
        target_dist = dists[game]
        for subsample_size in range(2, 11):
            sample_dist = get_sfp_dist(processed_data_path, df_payoff, data_type, 
                                       source, sample_id, subsample_size, False)
            score = stats.wasserstein_distance(sample_dist, target_dist)
            row = [sample_id, source, game, subsample_size, score]
            rows.append(row)

    df = pd.DataFrame(data=rows, columns=["sample", "source", "game", "subsample_size", "score"])
    save_loc = get_data_path(data_type, "images")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="game", y="score", hue="subsample_size", ax=ax)
    ax.set(ylim=(0, 0.7), ylabel="Earth Mover's Distance", xlabel="Game")
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/dists_comparison.png")
    plt.close()


def distances_plot_detailed(data_type, subsample_size):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = get_payoff_data(processed_data_path, data_type)
    samples = list(df_payoff[["sample", "source", "game"]].values)

    dists, games = get_theoretical_dists(1000)

    rows = []
    for sample_id, source, game in samples[0:1000]:
        sample_dist = get_sfp_dist(processed_data_path, df_payoff, data_type, 
                                   source, sample_id, subsample_size, False)
        for i in range(len(dists)):
            score = stats.wasserstein_distance(sample_dist, dists[i])
            row = [sample_id, source, game, games[i], score]
            rows.append(row)

    df = pd.DataFrame(data=rows, columns=["sample", "source", "game", "target", "score"])
    save_loc = get_data_path(data_type, "images")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="game", y="score", hue="target", palette=game_colors.values(), ax=ax)
    ax.set(ylim=(0, 0.7), ylabel="Earth Mover's Distance", xlabel="Game")
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/dists_comparison_{subsample_size}.png")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        distances_plot(sys.argv[1])
    elif len(sys.argv) == 3:
        distances_plot_detailed(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the data type and, optionally, the subsample size.")