import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from spatial_egt.common import game_colors, get_data_path, get_spatial_statistic_type


def plot_funcs(save_loc, file_name, df, label, stat_name, col):
    df = df.explode(stat_name)
    df["x"] = df.groupby(["source", "sample", label]).cumcount()
    col_order = None
    if label == "game":
        col_order = game_colors.keys()
        colors = game_colors.values()
    else:
        col_order = sorted(df[col].unique())
        colors = sns.color_palette("hls", len(df[col].unique()))
    facet = sns.FacetGrid(
        df,
        col=col,
        col_order=col_order,
        hue_order=col_order,
        palette=colors,
        height=4,
        aspect=1,
    )
    facet.map_dataframe(sns.lineplot, x="x", y=stat_name, errorbar="sd")
    facet.set_titles(col_template="{col_name}")
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def plot_dists(save_loc, file_name, df, label_name, stat_name, col):
    df = df.explode(stat_name).reset_index()
    order = None
    if label_name == "game":
        order = game_colors.keys()
    else:
        order = sorted(df[col].unique())
    facet = sns.FacetGrid(df, col=col, col_order=order, height=4, aspect=1, sharey=False)
    bins = np.histogram_bin_edges(df[stat_name].dropna())
    facet.map_dataframe(
        sns.histplot,
        x=stat_name,
        hue="sample",
        multiple="stack",
        bins=bins,
        edgecolor=None,
        stat="proportion",
        common_norm=False,
    )
    facet.set_titles(col_template="{col_name}")
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def plot_values(save_loc, file_name, df, label_name, stat_name, col):
    """https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/"""
    labels = sorted(df[col].unique())
    num_labels = len(labels)
    colors = sns.color_palette("hls", num_labels)
    gs = grid_spec.GridSpec(num_labels, 1)
    fig = plt.figure()
    axes = []
    min_val = df[stat_name].min()
    x = np.linspace(min_val, df[stat_name].max(), 100)
    for c, class_name in enumerate(labels):
        well_data = df.loc[df[col] == class_name][stat_name]
        kde = stats.gaussian_kde(well_data)
        axes.append(fig.add_subplot(gs[c : c + 1, 0:1]))
        axes[-1].plot(x, kde(x), color="white")
        axes[-1].fill_between(x, kde(x), alpha=0.75, color=colors[c])
        rect = axes[-1].patch
        rect.set_alpha(0)
        axes[-1].set_yticklabels([])
        if c == num_labels - 1:
            axes[-1].set_xlabel(stat_name)
        else:
            axes[-1].set(xticklabels=[], xticks=[])
        axes[-1].text(min_val - np.std(df[stat_name]) / 3, 0, class_name, ha="right")
        axes[-1].set(yticks=[])
        for s in ["top", "right", "left", "bottom"]:
            axes[-1].spines[s].set_visible(False)
    gs.update(hspace=-0.7)
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def get_data(df_stat, data_type, label_name, stat_name, source="", sample_ids=None):
    data_path = get_data_path(data_type, ".")
    df_labels = pd.read_csv(f"{data_path}/labels.csv")
    df_labels["sample"] = df_labels["sample"].astype(str)

    if source != "":
        df_labels = df_labels[df_labels["source"] == source]
    if sample_ids:
        df_labels = df_labels[df_labels["sample"].isin(sample_ids)]

    df_stat["sample"] = df_stat["sample"].astype(str)
    df = df_labels.merge(df_stat, on=["source", "sample"])
    df = df[["source", "sample", label_name, stat_name]]

    return df


def idv_plots(df_stat, data_type, time, label_name, stat_name, source, plot, *sample_ids):
    save_loc = get_data_path(data_type, f"images/{stat_name}", time)
    df = get_data(df_stat, data_type, label_name, stat_name, source=source, sample_ids=sample_ids)
    file_name = stat_name + "_" + source + "_" + "_".join(sample_ids)
    plot(save_loc, file_name, df, label_name, stat_name, "sample")


def agg_plot(df_stat, data_type, time, label_name, stat_name, source, plot):
    save_loc = get_data_path(data_type, f"images/{stat_name}", time)
    df = get_data(df_stat, data_type, label_name, stat_name, source=source)
    file_name = stat_name + source
    plot(save_loc, file_name, df, label_name, stat_name, label_name)


def main(data_type, time, label_name, stat_name, *filter_args):
    features_data_path = get_data_path(data_type, "statistics")
    df_stat = pd.read_pickle(f"{features_data_path}/{stat_name}.pkl")
    function_type = get_spatial_statistic_type(df_stat, stat_name)
    if function_type == "distribution":
        plot = plot_dists
    elif function_type == "function":
        plot = plot_funcs
    else:
        plot = plot_values

    if len(filter_args) == 0:
        agg_plot(df_stat, data_type, time, label_name, stat_name, "", plot)
    elif len(filter_args) == 1:
        source = filter_args[0]
        agg_plot(df_stat, data_type, time, label_name, stat_name, source, plot)
    elif len(filter_args) > 1:
        source = filter_args[0]
        sample_ids = filter_args[1:]
        idv_plots(df_stat, data_type, time, label_name, stat_name, source, plot, *sample_ids)


if __name__ == "__main__":
    if len(sys.argv) > 4:
        main(*sys.argv[1:])
    else:
        print("Please see the module docstring for usage instructions.")
