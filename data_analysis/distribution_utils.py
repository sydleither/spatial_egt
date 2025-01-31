import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import game_colors, get_data_path, read_payoff_df
from data_processing.spatial_statistics import (create_nc_dists, 
                                                create_sfp_dist, 
                                                create_subnc_dists,
                                                get_cell_type_counts)


sns.set_theme()
sns.set_style("white")


def plot_agg_dist(game_dists, save_loc, file_name, title, xlabel, ylabel):
    df_dict = {"data":[], "game":[]}
    for game in game_dists:
        game_fs_counts = game_dists[game]
        game_fs_counts = [x for y in game_fs_counts for x in y]
        df_dict["data"] += game_fs_counts
        df_dict["game"] += [game for _ in range(len(game_fs_counts))]
    
    df = pd.DataFrame(df_dict)
    facet = sns.FacetGrid(df, col="game", hue="game", 
                          col_order=game_colors.keys(), 
                          col_wrap=2, palette=game_colors.values(), 
                          height=4, aspect=1)
    facet.map_dataframe(sns.histplot, x="data", bins=10, kde=True,
                        kde_kws={"bw_adjust":2}, stat="proportion")
    facet.set_titles(col_template="{col_name}")
    facet.set(xlabel=xlabel, ylabel=ylabel)
    facet.figure.subplots_adjust(top=0.9)
    facet.figure.suptitle(title)
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def plot_idv_fs_count(dists, games, save_loc, file_name, title, xlabel, ylabel):
    fig, ax = plt.subplots(1, len(dists), figsize=(4*len(dists), 4))
    for a,sample_id in enumerate(dists):
        dist = dists[sample_id]
        game = games[sample_id]
        axis = ax[a] if len(dists) > 1 else ax
        axis.bar(range(len(dist)), dist, 
                  color=game_colors[game])
        axis.set(title=game, ylim=(0,1))
    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/{file_name}.png")


def plot_idv_dist(dists, games, save_loc, file_name, title, xlabel, ylabel):
    df_dict = {"data":[], "sample":[], "game":[]}
    for sample_id in dists:
        dist = dists[sample_id]
        game = games[sample_id]
        df_dict["data"] += dist
        df_dict["sample"] += [sample_id for _ in range(len(dist))]
        df_dict["game"] += [game for _ in range(len(dist))]
    
    df = pd.DataFrame(df_dict)
    facet = sns.FacetGrid(df, col="sample", hue="game", 
                          hue_order=game_colors.keys(), 
                          palette=game_colors.values(), 
                          height=4, aspect=1)
    facet.map_dataframe(sns.histplot, x="data", bins=10, kde=True,
                        kde_kws={"bw_adjust":2}, stat="proportion")
    facet.set_titles(col_template="{col_name}")
    facet.set(xlabel=xlabel, ylabel=ylabel)
    facet.figure.subplots_adjust(top=0.9)
    facet.figure.suptitle(title)
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def get_data(data_type, source, dist_func, limit=500):
    cnt = 0
    game_dists = {"sensitive_wins":[], "coexistence":[],
                     "bistability":[], "resistant_wins":[]}
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = read_payoff_df(processed_data_path)
    df_payoff = df_payoff[df_payoff["game"] != "unknown"]
    for sample_id in df_payoff["sample"].unique():
        file_name = f"spatial_{source}_{sample_id}.csv"
        df = pd.read_csv(f"{processed_data_path}/{file_name}")
        s, r = get_cell_type_counts(df)
        prop_s = s/(s+r)
        if prop_s < 0.05 or prop_s > 0.95:
            continue
        s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
        r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)
        if dist_func == "sfp":
            dist = create_sfp_dist(s_coords, r_coords, data_type)
        elif dist_func == "nc":
            dist, _ = create_nc_dists(s_coords, r_coords, data_type)
        elif dist_func == "subnc":
            dist, _ = create_subnc_dists(s_coords, r_coords, data_type)
        game = df_payoff.at[(sample_id, source), "game"]
        game_dists[game].append(dist)
        cnt += 1
        if cnt > limit:
            break
    return game_dists


def get_data_idv(data_type, source, dist_func, sample_ids):
    dists = dict()
    games = dict()
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = read_payoff_df(processed_data_path)
    df_payoff = df_payoff[df_payoff["game"] != "unknown"]
    for sample_id in sample_ids:
        file_name = f"spatial_{source}_{sample_id}.csv"
        df = pd.read_csv(f"{processed_data_path}/{file_name}")
        s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
        r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)
        if dist_func == "sfp":
            dist = create_sfp_dist(s_coords, r_coords, data_type)
        elif dist_func == "nc":
            dist, _ = create_nc_dists(s_coords, r_coords, data_type)
        elif dist_func == "subnc":
            dist, _ = create_subnc_dists(s_coords, r_coords, data_type)
        game = df_payoff.at[(sample_id, source), "game"]
        dists[sample_id] = dist
        games[sample_id] = game
    return dists, games
