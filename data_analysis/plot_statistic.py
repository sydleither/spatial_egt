import sys

import numpy as np
import pandas as pd
import seaborn as sns

from spatial_egt.common import game_colors, get_data_path, get_spatial_statistic_type


sns.set_theme()
sns.set_style("white")


def plot_funcs(save_loc, file_name, df, label_name, stat_name):
    df = df.explode(stat_name)
    df["x"] = df.groupby(["source", "sample", label_name]).cumcount()
    order = None
    if label_name == "game":
        order = game_colors.keys()
        palette = game_colors.values()
    else:
        order = sorted(df[label_name].unique())
        palette = sns.color_palette("hls", len(df[label_name].unique()))
    facet = sns.FacetGrid(df, col=label_name, hue=label_name,
                          col_order=order, hue_order=order,
                          palette=palette, height=4, aspect=1)
    facet.map_dataframe(sns.lineplot, x="x", y=stat_name, errorbar="sd")
    facet.set_titles(col_template="{col_name}")
    facet.figure.subplots_adjust(top=0.9)
    facet.figure.suptitle(stat_name.replace("_", " "))
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def plot_dists(save_loc, file_name, df, label_name, stat_name):
    df = df.explode(stat_name)
    order = None
    if label_name == "game":
        order = game_colors.keys()
    else:
        order = sorted(df[label_name].unique())
    facet = sns.FacetGrid(df, col=label_name, hue="sample",
                          col_order=order, height=4, aspect=1)
    facet.map_dataframe(sns.kdeplot, x=stat_name)
    facet.set(xlim=(df[stat_name].min(), df[stat_name].max()))
    facet.set_titles(col_template="{col_name}")
    facet.figure.subplots_adjust(top=0.9)
    facet.figure.suptitle(stat_name.replace("_", " "))
    facet.tight_layout()
    facet.figure.patch.set_alpha(0.0)
    facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


def plot_values():
    pass


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


def idv_plots(df_stat, data_type, label_name, stat_name, source, plot, *sample_ids):
    save_loc = get_data_path(data_type, f"images/{stat_name}")
    df = get_data(df_stat, data_type, label_name, stat_name, source=source, sample_ids=sample_ids)
    file_name = stat_name+"_"+source+"_"+"_".join(sample_ids)
    plot(save_loc, file_name, df, label_name, stat_name)


def agg_plot(df_stat, data_type, label_name, stat_name, source, plot):
    save_loc = get_data_path(data_type, f"images/{stat_name}")
    df = get_data(df_stat, data_type, label_name, stat_name, source=source)
    file_name = stat_name+source
    plot(save_loc, file_name, df, label_name, stat_name)


def main(data_type, label_name, stat_name, *filter_args):
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
        agg_plot(df_stat, data_type, label_name, stat_name, "", plot)
    elif len(filter_args) == 1:
        source = filter_args[0]
        agg_plot(df_stat, data_type, label_name, stat_name, source, plot)
    elif len(filter_args) > 5:
        source = filter_args[0]
        sample_ids = filter_args[1:]
        idv_plots(df_stat, data_type, label_name, stat_name, source, plot, *sample_ids)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(*sys.argv[1:])
    else:
        print("Please see the module docstring for usage instructions.")
