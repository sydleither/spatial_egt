import sys

import pandas as pd
import numpy as np
import seaborn as sns

from common import game_colors, get_data_path, read_payoff_df
from data_processing.spatial_statistics import (create_cpcf,
                                                create_muspan_domain,
                                                create_nc_dists,
                                                create_nn_dist,
                                                create_pcf,
                                                create_ripleysk,
                                                create_sfp_dist,
                                                get_dist_params)


sns.set_theme()
sns.set_style("white")


def plot_lines(dists, games, save_loc, file_name, title, xlabel, ylabel, col="game"):
    df_dict = {"x":[], "y":[], "sample":[], "game":[]}
    for sample_id in dists:
        dist = dists[sample_id]
        game = games[sample_id]
        df_dict["y"] += list(dist)
        df_dict["x"] += [j for j in range(len(dist))]
        df_dict["sample"] += [sample_id for _ in range(len(dist))]
        df_dict["game"] += [game for _ in range(len(dist))]
    
    df = pd.DataFrame(df_dict)
    facet = sns.FacetGrid(df, col=col, hue="game",
                          hue_order=game_colors.keys(),
                          palette=game_colors.values(),
                          height=4, aspect=1)
    facet.map_dataframe(sns.lineplot, x="x", y="y")
    facet.set_titles(col_template="{col_name}")
    facet.set(xlabel=xlabel, ylabel=ylabel)
    facet.figure.subplots_adjust(top=0.9)
    facet.figure.suptitle(title)
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def plot_dists(dists, games, save_loc, file_name, title, xlabel, ylabel, col="game"):
    df_dict = {"data":[], "sample":[], "game":[]}
    for sample_id in dists:
        dist = dists[sample_id]
        game = games[sample_id]
        df_dict["data"] += dist
        df_dict["sample"] += [sample_id for _ in range(len(dist))]
        df_dict["game"] += [game for _ in range(len(dist))]
    
    bins = np.arange(0, 1.01, 0.01)
    df = pd.DataFrame(df_dict)
    facet = sns.FacetGrid(df, col=col, hue="game",
                          hue_order=game_colors.keys(),
                          palette=game_colors.values(),
                          height=4, aspect=1)
    facet.map_dataframe(sns.histplot, x="data", bins=bins, kde=True,
                        kde_kws={"bw_adjust":2}, stat="proportion")
    facet.set_titles(col_template="{col_name}")
    facet.set(xlabel=xlabel, ylabel=ylabel)
    facet.figure.subplots_adjust(top=0.9)
    facet.figure.suptitle(title)
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def get_data(data_type, dist_func, source="", sample_ids=None, limit=500):
    dists = dict()
    games = dict()
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = read_payoff_df(processed_data_path)
    df_payoff = df_payoff[df_payoff["game"] != "unknown"]
    if not source == "":
        df_payoff = df_payoff[df_payoff["source"] == source]
    if sample_ids:
        df_payoff = df_payoff[df_payoff["sample"].isin(sample_ids)]
    cnt = 0
    for (source, sample_id) in df_payoff[["source", "sample"]].values:
        file_name = f"{source} {sample_id}.csv"
        df = pd.read_csv(f"{processed_data_path}/{file_name}")
        dimensions = list(df.drop("type", axis=1).columns)
        params = get_dist_params(data_type, dimensions)[dist_func]
        s_coords_df = df.loc[df["type"] == "sensitive"][dimensions]
        s_coords = list(s_coords_df.values)
        r_coords = list(df.loc[df["type"] == "resistant"][dimensions].values)
        if dist_func == "sfp":
            dist = create_sfp_dist(s_coords, r_coords, params["sample_length"])
        elif dist_func == "nc":
            dist, _ = create_nc_dists(s_coords, r_coords, params["radius"])
        elif dist_func == "pcf":
            dist = create_pcf(s_coords_df, params["max_r"], params["dr"], dimensions)
        elif dist_func == "rk":
            dist = create_ripleysk(s_coords, params["boundary"], dimensions)
        elif dist_func == "cpcf":
            domain = create_muspan_domain(df)
            dist = create_cpcf(domain, "sensitive", "resistant",
                               params["max_radius"],
                               params["annulus_step"],
                               params["annulus_width"])
        elif dist_func == "nn":
            domain = create_muspan_domain(df)
            dist = create_nn_dist(domain, "sensitive", "resistant")
        game = df_payoff.at[(source, sample_id), "game"]
        dists[sample_id] = dist
        games[sample_id] = game
        if cnt > limit:
            break
        cnt += 1
    return dists, games


def idv_plots(dist, title, xlabel, ylabel, plot_func, data_type, source, *sample_ids):
    save_loc = get_data_path(data_type, "images")
    dists, games = get_data(data_type, dist, source=source, sample_ids=sample_ids)
    file_name = dist+"_"+source+"_"+"_".join(sample_ids)
    plot_func(dists, games, save_loc, file_name, title, xlabel, ylabel, "sample")


def agg_plot(dist, title, xlabel, ylabel, plot_func, data_type, source):
    save_loc = get_data_path(data_type, "images")
    dists, games = get_data(data_type, dist, source=source)
    plot_func(dists, games, save_loc, dist+source, title, xlabel, ylabel, "game")


def main():
    dist = sys.argv[1]
    if dist == "sfp":
        title = "Spatial Subsampling Distribution"
        xlabel = "Fraction Sensitive"
        ylabel = "Frequency Across Subsamples"
        plot_func = plot_dists
    elif dist == "nc":
        title = "Neighborhood Composition Distribution"
        xlabel = "Fraction Sensitive in Neighborhood"
        ylabel = "Fraction of Resistant Cells"
        plot_func = plot_dists
    elif dist == "pcf":
        title = "S Pair Correlation"
        xlabel = "r"
        ylabel = "g(r)"
        plot_func = plot_lines
    elif dist == "cpcf":
        title = "SR Pair Correlation"
        xlabel = "r"
        ylabel = "g(r)"
        plot_func = plot_lines
    elif dist == "nn":
        title = "Nearest Neighbor Distribution"
        xlabel = "Distance"
        ylabel = "Proportion"
        plot_func = plot_lines

    if len(sys.argv) == 3:
        agg_plot(dist, title, xlabel, ylabel, plot_func, sys.argv[2], "")
    elif len(sys.argv) == 4:
        agg_plot(dist, title, xlabel, ylabel, plot_func, sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 4:
        idv_plots(dist, title, xlabel, ylabel, plot_func, sys.argv[2], sys.argv[3], *sys.argv[4:])


if __name__ == "__main__":
    main()
