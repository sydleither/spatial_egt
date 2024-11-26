from collections import Counter, OrderedDict
import os
from random import sample

import matplotlib.pyplot as plt
import pandas as pd

from common import game_colors, get_data_path
from data_processing.processed_to_features import read_processed_sample
from data_processing.spatial_statistics import (calculate_game,
                                                create_nc_dists, 
                                                create_sfp_dist, 
                                                get_cell_type_counts)


def plot_agg_dist(game_dists, save_loc, file_name, title, xlabel, ylabel, rnd):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for game in game_dists.keys():
        game_fs_counts = game_dists[game]
        game_fs_counts = [x for y in game_fs_counts for x in y]
        game_fs_counts = [round(x, rnd) for x in game_fs_counts]
        a = 0 if "wins" in game else 1
        counts = Counter(game_fs_counts)
        y_sum = sum(counts.values())
        freqs = [y/y_sum for y in counts.values()]
        ax[a].bar(counts.keys(), freqs, width=10**(-rnd), 
                  label=game, color=game_colors[game], alpha=0.66)
    ax[0].legend()
    ax[1].legend()
    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/{file_name}.png")


def plot_idv_fs_count(dists, games, save_loc, file_name, title, xlabel, ylabel):
    fig, ax = plt.subplots(1, len(dists), figsize=(4*len(dists), 4))
    for a,sample_id in enumerate(dists):
        dist = dists[sample_id]
        game = games[sample_id]
        ax[a].bar(range(len(dist)), sorted(dist), 
                  color=game_colors[game])
        ax[a].set(title=game, ylim=(0,1))
    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/{file_name}.png")


def plot_idv_dist(dists, games, save_loc, file_name, title, xlabel, ylabel, rnd):
    fig, ax = plt.subplots(1, len(dists), figsize=(4*len(dists), 4))
    for a,sample_id in enumerate(dists):
        dist = dists[sample_id]
        game = games[sample_id]
        dist = [round(x, rnd) for x in dist]
        counts = Counter(dist)
        counts = OrderedDict(sorted(counts.items()))
        y_sum = sum(counts.values())
        freqs = [y/y_sum for y in counts.values()]
        ax[a].bar(counts.keys(), freqs, width=10**(-rnd),
                  label=game, color=game_colors[game], alpha=0.66)
        ax[a].set(title=f"{game}\n{sample_id}", xlim=(0,1), ylim=(0,1))
    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/{file_name}.png")


def get_data(data_type, dist_func, limit=500):
    cnt = 0
    game_dists = {"sensitive_wins":[], "coexistence":[],
                     "bistability":[], "resistant_wins":[]}
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    df_payoff["game"] = df_payoff.apply(calculate_game, axis="columns")
    df_payoff = df_payoff[df_payoff["game"] != "unknown"]
    for sample_id in df_payoff["sample"].unique():
        file_name = f"spatial_HAL_{sample_id}.csv"
        df = read_processed_sample(processed_data_path,
                                   file_name, df_payoff)
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
        game = df_payoff[df_payoff["sample"] == sample_id]["game"].iloc[0]
        game_dists[game].append(dist)
        cnt += 1
        if cnt > limit:
            break
    return game_dists


def get_data_idv(data_type, dist_func, sample_ids):
    dists = dict()
    games = dict()
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    df_payoff["game"] = df_payoff.apply(calculate_game, axis="columns")
    for sample_id in sample_ids:
        file_name = f"spatial_HAL_{sample_id}.csv"
        df = read_processed_sample(processed_data_path,
                                   file_name, df_payoff)
        s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
        r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)
        if dist_func == "sfp":
            dist = create_sfp_dist(s_coords, r_coords)
        elif dist_func == "nc":
            dist, _ = create_nc_dists(s_coords, r_coords, data_type)
        game = df_payoff[df_payoff["sample"] == sample_id]["game"].iloc[0]
        dists[sample_id] = dist
        games[sample_id] = game
    return dists, games