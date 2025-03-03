import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
import seaborn as sns

from common import game_colors, get_data_path
from data_analysis.plot_function import get_data
from data_processing.write_feature_jobs import DISTRIBUTION_BINS


def bin_samples(function_name, samples):
    start, stop, step = DISTRIBUTION_BINS[function_name]
    bins = np.arange(start, stop, step)
    binned_samples = []
    for data, game in samples:
        binned_data, _ = np.histogram(data, bins=bins)
        binned_samples.append([binned_data, game])
    return binned_samples


def get_difference_function(function1, function2):
    return euclidean(function1, function2)


def get_difference_distribution(distribution1, distribution2):
    return wasserstein_distance(distribution1, distribution2)


def get_difference_feature(feature1, feature2):
    return abs(feature1 - feature2)


def plot_heatmap(save_loc, feature, df, stat):
    order = game_colors.keys()
    df = df.groupby(["Game 1", "Game 2"]).mean().reset_index()
    df = df.pivot(index="Game 1", columns="Game 2", values=stat)
    df = df.reindex(order)
    df = df[order]
    fig, ax = plt.subplots()
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Greens", ax=ax)
    ax.set(xlabel="Game", ylabel="Game")
    feature_title = feature.replace("_", " ")
    ax.set(title=f"{feature_title} {stat} Between Games")
    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/distinguish_hm_{feature}.png", bbox_inches="tight")


def plot_barplot(save_loc, feature, df, stat):
    order = game_colors.keys()
    colors = game_colors.values()
    fig, ax = plt.subplots()
    bp = sns.barplot(data=df, x="Game 1", y=stat, hue="Game 2", ax=ax,
                     order=order, hue_order=order, palette=colors)
    ax.set(xlabel="Game")
    bp.legend_.set_title("Game")
    feature_title = feature.replace("_", " ")
    ax.set(title=f"{feature_title} {stat} Between Games")
    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/distinguish_bp_{feature}.png", bbox_inches="tight")


def get_binary_pct_diff(df, stat):
    df["same"] = df["Game 1"] == df["Game 2"]
    df = df.drop_duplicates(subset=["i", "j"], keep="first")
    df = df[[stat, "same"]].groupby("same").mean()
    d = df.at[False, stat]
    s = df.at[True, stat]
    return abs((s-d)/((s+d)/2))


def get_difference_data(data_type, features_data_path, feature_name, df_features, limit=500):
    if feature_name in df_features.columns:
        samples = df_features[[feature_name, "game"]].values
        get_difference = get_difference_feature
        stat = "Difference"
    else:
        df_func = pd.read_pickle(f"{features_data_path}/{feature_name}.pkl")
        samples = get_data(df_func, feature_name, data_type)[[feature_name, "game"]].values
        if df_func["type"].iloc[0] == "distribution":
            samples = bin_samples(feature_name, samples)
            get_difference = get_difference_distribution
            stat = "Wasserstein Distance"
        else:
            get_difference = get_difference_function
            stat = "Euclidean Distance"

    data = []
    samples = samples[0:limit]
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            feature_i, game_i = samples[i]
            feature_j, game_j = samples[j]
            diff = get_difference(feature_i, feature_j)
            data.append({"i":i, "j":j, "Game 1":game_i, "Game 2":game_j, stat:diff})
            if game_i != game_j:
                data.append({"i":i, "j":j, "Game 1":game_j, "Game 2":game_i, stat:diff})

    return pd.DataFrame(data), stat


def main_idv(data_type, feature_name):
    features_data_path = get_data_path(data_type, "features")
    df_features = pd.read_csv(f"{features_data_path}/all.csv")
    df_features = df_features[df_features["game"] != "Unknown"]

    df, stat = get_difference_data(data_type, features_data_path, feature_name, df_features)

    save_loc = get_data_path(data_type, "images")
    plot_heatmap(save_loc, feature_name, df, stat)
    plot_barplot(save_loc, feature_name, df, stat)


def main_all(data_type):
    features_data_path = get_data_path(data_type, "features")
    df_features = pd.read_csv(f"{features_data_path}/all.csv")
    df_features = df_features[df_features["game"] != "Unknown"]

    pct_diff_data = []
    for feature_file in os.listdir(features_data_path):
        if feature_file == "all.csv":
            continue
        feature_name = feature_file[:-4]
        df, stat = get_difference_data(data_type, features_data_path, feature_name, df_features, 50)
        pct_diff = get_binary_pct_diff(df, stat)
        display_name = feature_name.replace("_", " ")
        pct_diff_data.append({"Spatial Statistic":display_name, "Pct Difference":pct_diff})

    save_loc = get_data_path(data_type, "images")
    df = pd.DataFrame(pct_diff_data).sort_values("Pct Difference", ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="Spatial Statistic", y="Pct Difference", ax=ax)
    ax.set(title="Percent Difference in Difference Statistic Across Spatial Statistics")
    ax.tick_params(axis="x", labelrotation=90)
    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/distinguish_all.png", bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main_all(sys.argv[1])
    elif len(sys.argv) == 3:
        main_idv(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the data type and, optionally, a feature name.")
