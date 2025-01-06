import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

from common.common import game_colors, get_data_path
from common.distributions import (fit_beta, get_payoff_data, 
                                  get_sfp_dist,
                                  get_theoretical_dists)


def plot_dists(save_loc, dists, games, n):
    freq = np.ones_like(range(n))/n
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(len(dists)):
        ax[i].hist(dists[i], bins=20, weights=freq, color=game_colors[games[i]])
        ax[i].set(xlim=(0, 1), ylim=(0, 0.2))
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/dists.png")
    plt.close()


def distances_plot(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = get_payoff_data(processed_data_path)
    samples = list(df_payoff[["sample", "source", "game"]].values)

    dists, games = get_theoretical_dists(1000)
    dists = {games[i]:dists[i] for i in range(len(games))}
    sizes = range(2,11) if data_type == "in_silico" else range(20,110,10)

    rows = []
    for sample_id, source, game in samples:
        target_dist = dists[game]
        for subsample_size in sizes:
            sample_dist = get_sfp_dist(processed_data_path, df_payoff, data_type, 
                                       source, sample_id, subsample_size, False)
            score = stats.wasserstein_distance(sample_dist, target_dist)
            row = [sample_id, source, game, subsample_size, score]
            rows.append(row)

    df = pd.DataFrame(data=rows, columns=["sample", "source", "game", "subsample_size", "score"])
    save_loc = get_data_path(data_type, "images")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="game", y="score", hue="subsample_size", 
                order=list(game_colors.keys()), ax=ax)
    ax.set(ylabel="Earth Mover's Distance", xlabel="Game")
    ax.get_legend().remove()
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/dists_comparison.png")
    plt.close()


def distances_plot_fit(data_type, subsample_size):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = get_payoff_data(processed_data_path)
    samples = list(df_payoff[["sample", "source", "game"]].values)

    rows = []
    for sample_id, source, game in samples[0:1000]:
        sample_dist = get_sfp_dist(processed_data_path, df_payoff, data_type, 
                                   source, sample_id, int(subsample_size), False)
        a, b, mean, var = fit_beta(sample_dist)
        rows.append([sample_id, source, game, a, b, mean, var])

    df = pd.DataFrame(data=rows, columns=["sample", "source", "game",
                                          "a", "b", "mean", "var"])
    save_loc = get_data_path(data_type, "images")
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        game = list(game_colors.keys())[i]
        sns.histplot(data=df[df["game"] == game], x="a", y="b",
                    color=game_colors[game], ax=ax[i])
        ax[i].set(title=game, xlim=(0,6), ylim=(0,6))
    #ax.set(ylim=(0, 0.7), ylabel="Earth Mover's Distance", xlabel="Game")
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/dists_comparison_{subsample_size}.png")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        distances_plot(sys.argv[1])
    elif len(sys.argv) == 3:
        distances_plot_fit(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the data type and, optionally, the subsample size.")