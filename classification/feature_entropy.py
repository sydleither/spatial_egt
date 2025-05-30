from itertools import combinations
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

from spatial_egt.classification.common import get_feature_data
from spatial_egt.classification.DDIT.DDIT import DDIT
from spatial_egt.classification.feature_plot_utils import format_df, plot_feature_selection


def joint_entropy_plot(save_loc, df):
    # format so heatmap is symmetric
    df_temp = df.copy()
    df_temp["Feature 0"] = df_temp["Feature"]
    df_temp["Feature"] = df_temp["Feature 1"]
    df_temp["Feature 1"] = df_temp["Feature 0"]
    df_temp = df_temp.drop("Feature 0", axis=1)
    df = pd.concat([df, df_temp])

    # format feature names
    df["Feature"] = df["Feature"].str.replace("_", " ")
    df["Feature 1"] = df["Feature 1"].str.replace("_", " ")

    # create joint heatmap and sort by mean value
    df_hm = df.pivot(index="Feature", columns="Feature 1", values="Interaction Information")
    means = df_hm.mean().sort_values(ascending=False)
    df_hm = df_hm.reindex(means.index, axis=0).reindex(means.index, axis=1)

    # set colorbar to diverge at 0
    min_entropy = df["Interaction Information"].min()
    max_entropy = df["Interaction Information"].max()
    if min_entropy < 0:
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=min_entropy, vmax=max_entropy)
        cmap = plt.get_cmap("PuOr")
    else:
        norm = mcolors.Normalize(vmin=min_entropy, vmax=max_entropy)
        cmap = plt.get_cmap("Purples")
    bar_colors = cmap(norm(means))

    # plot
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 4], width_ratios=[4, 0.25], hspace=0.05, wspace=0.05
    )
    ax_hm = fig.add_subplot(gs[1, 0])
    cbar_ax = fig.add_subplot(gs[1, 1])
    sns.heatmap(
        df_hm,
        cmap=cmap,
        norm=norm,
        ax=ax_hm,
        cbar=True,
        cbar_ax=cbar_ax,
        xticklabels=True,
        yticklabels=True,
    )
    ax_hm.set(xlabel=None, ylabel=None)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_hm)
    ax_top.bar(range(len(df_hm)), means, width=0.9, align="edge", color=bar_colors)
    ax_top.axis("off")
    ax_top.set(title="Interaction Information")
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/interaction_information.png", bbox_inches="tight")
    plt.close()


def fragmentation_data(df, feature_names, label_name, order, value_label):
    # initializations
    ddit = DDIT()

    # bin and register features
    nbins = int(np.ceil(np.log2(len(df)) + 1))
    for feature_name in feature_names:
        column_data = df[feature_name].values
        binned_column_data = pd.qcut(column_data, nbins, labels=False, duplicates="drop")
        ddit.register_column_tuple(feature_name, tuple(binned_column_data))
    ddit.register_column_tuple(label_name, tuple(df[label_name].values))

    # calculate entropies
    feature_sets = combinations(feature_names, order)
    label_entropy = ddit.H(label_name)
    results = []
    for feature_set in feature_sets:
        ent = (
            ddit.recursively_solve_formula(label_name + ":" + "&".join(feature_set)) / label_entropy
        )
        feature_0 = {"Feature": feature_set[0]}
        feature_rest = {f"Feature {i}": feature_set[i] for i in range(1, order)}
        results.append(feature_0 | feature_rest | {value_label: float(ent)})

    return pd.DataFrame(results)


def main(data_type, label_name, feature_names):
    # get feature data
    save_loc, df_org, feature_names = get_feature_data(
        data_type, label_name, feature_names, "entropy"
    )
    feature_names = sorted(feature_names)
    feature_df = df_org[feature_names + [label_name]]
    value_label = "Mutual Information"

    # run and plot single-feature shared entropies
    df1 = fragmentation_data(feature_df, feature_names, label_name, 1, value_label)
    print(df1.sort_values(value_label))
    df1_formatted = format_df(df1.sort_values(value_label, ascending=False))
    plot_feature_selection(save_loc, value_label, None, df1_formatted)

    # get sum of entropy for each combination of features
    df_diff = df1.merge(df1, how="cross", suffixes=("", " 1"))
    df_diff["Entropy Sum"] = df_diff[value_label + ""] + df_diff[value_label + " 1"]
    df_diff = df_diff[["Feature", "Feature 1", "Entropy Sum"]]

    # calculate triple information
    df2 = fragmentation_data(feature_df, feature_names, label_name, 2, value_label)
    df = df2.merge(df_diff, on=["Feature", "Feature 1"])
    df["Interaction Information"] = df[value_label] - df["Entropy Sum"]

    # plot
    joint_entropy_plot(save_loc, df)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        print("Please provide the data type, label name, and feature set/names.")
