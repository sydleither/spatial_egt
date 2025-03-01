import sys

import muspan as ms
import pandas as pd
import seaborn as sns

from common import game_colors, get_data_path, read_payoff_df
from data_processing.spatial_statistics import (create_cpcf,
                                                create_moransi,
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
        df_dict["data"] += list(dist)
        df_dict["sample"] += [sample_id for _ in range(len(dist))]
        df_dict["game"] += [game for _ in range(len(dist))]
    
    bins = 10
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


def get_dist_by_name(dist_name, data_type, df):
    dimensions = list(df.drop("type", axis=1).columns)
    params = get_dist_params(data_type, dimensions)
    s_coords_df = df.loc[df["type"] == "sensitive"][dimensions]
    s_coords = list(s_coords_df.values)
    r_coords = list(df.loc[df["type"] == "resistant"][dimensions].values)
    dist = None
    if dist_name == "sfp":
        dist = create_sfp_dist(s_coords, r_coords, params[dist_name]["sample_length"])
    elif dist_name == "nc":
        dist, _ = create_nc_dists(s_coords, r_coords, params[dist_name]["radius"])
    elif dist_name == "pcf":
        dist = create_pcf(s_coords_df, params[dist_name]["max_r"], params[dist_name]["dr"], dimensions)
    elif dist_name == "rk":
        dist = create_ripleysk(s_coords, params[dist_name]["boundary"], dimensions)
    elif dist_name == "cpcf":
        domain = create_muspan_domain(df, dimensions)
        s_cells = ms.query.query(domain, ("label", "type"), "is", "sensitive")
        r_cells = ms.query.query(domain, ("label", "type"), "is", "resistant")
        dist = create_cpcf(domain, s_cells, r_cells,
                            params[dist_name]["max_radius"],
                            params[dist_name]["annulus_step"],
                            params[dist_name]["annulus_width"])
    elif dist_name == "nn":
        domain = create_muspan_domain(df, dimensions)
        s_cells = ms.query.query(domain, ("label", "type"), "is", "sensitive")
        r_cells = ms.query.query(domain, ("label", "type"), "is", "resistant")
        dist = create_nn_dist(domain, s_cells, r_cells)
    elif dist_name == "moransi":
        domain = create_muspan_domain(df, dimensions)
        dist = create_moransi(domain, "sensitive")
    return dist


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
        dist = get_dist_by_name(dist_func, data_type, df)
        game = df_payoff.at[(source, sample_id), "game"]
        dists[sample_id] = dist
        games[sample_id] = game
        cnt += 1
        if cnt >= limit:
            break
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
        plot_func = plot_dists

    if len(sys.argv) == 3:
        agg_plot(dist, title, xlabel, ylabel, plot_func, sys.argv[2], "")
    elif len(sys.argv) == 4:
        agg_plot(dist, title, xlabel, ylabel, plot_func, sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 4:
        idv_plots(dist, title, xlabel, ylabel, plot_func, sys.argv[2], sys.argv[3], *sys.argv[4:])


if __name__ == "__main__":
    main()
