import sys

import numpy as np
import pandas as pd
import seaborn as sns

from common import game_colors, get_data_path
from data_processing.write_feature_jobs import DISTRIBUTION_BINS, FUNCTION_LABELS


sns.set_theme()
sns.set_style("white")


def plot_funcs(df, func_name, save_loc, file_name, title, xlabel, ylabel, col="game"):
    df["x"] = df.groupby(["source", "sample", "game"]).cumcount()
    order = None
    if col == "game":
        order = game_colors.keys()
    facet = sns.FacetGrid(df, col=col, hue="game",
                          col_order=order, hue_order=game_colors.keys(),
                          palette=game_colors.values(), height=4, aspect=1)
    facet.map_dataframe(sns.lineplot, x="x", y=func_name)
    facet.set_titles(col_template="{col_name}")
    facet.set(xlabel=xlabel, ylabel=ylabel)
    facet.figure.subplots_adjust(top=0.9)
    facet.figure.suptitle(title)
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def plot_dists(df, dist_name, save_loc, file_name, title, xlabel, ylabel, col="game"):
    start, stop, step = DISTRIBUTION_BINS[dist_name]
    bins = np.arange(start, stop, step)
    order = None
    if col == "game":
        order = game_colors.keys()
    facet = sns.FacetGrid(df, col=col, hue="game",
                          col_order=order, hue_order=game_colors.keys(),
                          palette=game_colors.values(), height=4, aspect=1)
    facet.map_dataframe(sns.histplot, x=dist_name, bins=bins, kde=True,
                        kde_kws={"bw_adjust":2}, stat="proportion")
    facet.set_titles(col_template="{col_name}")
    facet.set(xlabel=xlabel, ylabel=ylabel)
    facet.figure.subplots_adjust(top=0.9)
    facet.figure.suptitle(title)
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def get_data(df_func, func_name, data_type, source="", sample_ids=None):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    df_payoff = df_payoff[df_payoff["game"] != "Unknown"]

    if not source == "":
        df_payoff = df_payoff[df_payoff["source"] == source]
    if sample_ids:
        df_payoff = df_payoff[df_payoff["sample"].isin(sample_ids)]

    df_func["sample"] = df_func["sample"].astype(str)
    df = df_payoff.merge(df_func, on=["source", "sample"])
    df = df[["source", "sample", "game", func_name]]
    
    return df


def idv_plots(df_func, func_name, title, xlabel, ylabel, plot, data_type, source, *sample_ids):
    save_loc = get_data_path(data_type, "images")
    df = get_data(df_func, func_name, data_type, source=source, sample_ids=sample_ids)
    df = df.explode(func_name)
    file_name = func_name+"_"+source+"_"+"_".join(sample_ids)
    plot(df, func_name, save_loc, file_name, title, xlabel, ylabel, "sample")


def agg_plot(df_func, func_name, title, xlabel, ylabel, plot, data_type, source):
    save_loc = get_data_path(data_type, "images")
    df = get_data(df_func, func_name, data_type, source=source)
    df = df.explode(func_name)
    plot(df, func_name, save_loc, func_name+source, title, xlabel, ylabel, "game")


def main():
    data_type = sys.argv[1]
    func_name = sys.argv[2]

    features_data_path = get_data_path(data_type, "features")
    df_func = pd.read_pickle(f"{features_data_path}/{func_name}.pkl")
    function_type = df_func["type"].iloc[0]
    if function_type == "distribution":
        plot = plot_dists
    elif function_type == "function":
        plot = plot_funcs
    else:
        return

    xlabel = FUNCTION_LABELS[func_name]["x"]
    ylabel = FUNCTION_LABELS[func_name]["y"]
    title = func_name.replace("_", " ")

    if len(sys.argv) == 3:
        agg_plot(df_func, func_name, title, xlabel, ylabel, plot, data_type, "")
    elif len(sys.argv) == 4:
        source = sys.argv[3]
        agg_plot(df_func, func_name, title, xlabel, ylabel, plot, data_type, source)
    elif len(sys.argv) > 4:
        source = sys.argv[3]
        sample_ids = sys.argv[4:]
        idv_plots(df_func, func_name, title, xlabel, ylabel, plot, data_type, source, *sample_ids)


if __name__ == "__main__":
    main()
