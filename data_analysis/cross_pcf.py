import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import get_data_path, game_colors
from data_analysis.distribution_utils import get_data, get_data_idv


def plot_agg_dist(game_dists, save_loc, file_name, title, xlabel, ylabel):
    df_dict = {"r":[], "gr":[], "game":[], "sample":[]}
    for game in game_dists:
        dists = game_dists[game]
        for i,dist in enumerate(dists):
            df_dict["gr"] += list(dist)
            df_dict["game"] += [game for _ in range(len(dist))]
            df_dict["sample"] += [i for _ in range(len(dist))]
            df_dict["r"] += [j for j in range(len(1,dist+1))]
    
    df = pd.DataFrame(df_dict)
    facet = sns.FacetGrid(df, col="game", hue="game", 
                          col_order=game_colors.keys(), 
                          col_wrap=2, palette=game_colors.values(), 
                          height=4, aspect=1)
    facet.map_dataframe(sns.lineplot, x="r", y="gr")
    facet.set_titles(col_template="{col_name}")
    facet.set(xlabel=xlabel, ylabel=ylabel)
    facet.figure.subplots_adjust(top=0.9)
    facet.figure.suptitle(title)
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


# def plot_idv_dist(dists, games, save_loc, file_name, title, xlabel, ylabel):
#     df_dict = {"data":[], "sample":[], "game":[]}
#     for sample_id in dists:
#         dist = dists[sample_id]
#         game = games[sample_id]
#         df_dict["data"] += dist
#         df_dict["sample"] += [sample_id for _ in range(len(dist))]
#         df_dict["game"] += [game for _ in range(len(dist))]
    
#     df = pd.DataFrame(df_dict)
#     facet = sns.FacetGrid(df, col="sample", hue="game", 
#                           hue_order=game_colors.keys(), 
#                           palette=game_colors.values(), 
#                           height=4, aspect=1)
#     facet.map_dataframe(sns.histplot, x="data", bins=10, kde=True,
#                         kde_kws={"bw_adjust":2}, stat="proportion")
#     facet.set_titles(col_template="{col_name}")
#     facet.set(xlabel=xlabel, ylabel=ylabel)
#     facet.figure.subplots_adjust(top=0.9)
#     facet.figure.suptitle(title)
#     facet.tight_layout()
#     facet.figure.patch.set_alpha(0.0)
#     facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


# def idv_plots(data_type, source, *sample_ids):
#     save_loc = get_data_path(data_type, "images")
#     dists, games = get_data_idv(data_type, source, "pcf", sample_ids)
#     plot_idv_dist(dists, games, save_loc,
#                   source+"_pcf_"+"_".join(sample_ids),
#                   "SR Pair Correlations", "r", "g(r)")


def agg_plot(data_type, source):
    save_loc = get_data_path(data_type, "images")
    n = 500
    fs_dists = get_data(data_type, source, "pcf", n)
    plot_agg_dist(fs_dists, save_loc, source+"_pcf",
                  f"SR Pair Correlations\n{n} samples",
                  "r", "g(r)")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        agg_plot(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3:
        print()
        # idv_plots(sys.argv[1], sys.argv[2], *sys.argv[3:])
    else:
        print("Please provide the data type and source.")