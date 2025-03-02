import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import game_colors, get_data_path
from data_analysis.plot_function import get_data
from data_processing.write_feature_jobs import DISTRIBUTION_BINS


def get_functions(features_data_path, function_name, data_type):
    df_func = pd.read_pickle(f"{features_data_path}/{function_name}.pkl")
    samples = get_data(df_func, function_name, data_type)[[function_name, "game"]].values

    function_type = df_func["type"].iloc[0]
    if function_type == "distribution":
        start, stop, step = DISTRIBUTION_BINS[function_name]
        bins = np.arange(start, stop, step)
        binned_samples = []
        for data, game in samples:
            binned_data, _ = np.histogram(data, bins=bins)
            binned_samples.append([binned_data, game])
        return binned_samples

    return samples


def get_difference_function(function1, function2):
    return 0


def get_difference_feature(feature1, feature2):
    return abs(feature1 - feature2)


def plot_heatmap(save_loc, feature, df, stat):
    order = game_colors.keys()
    df = df.groupby(["Game 1", "Game 2"]).mean().reset_index()
    df = df.pivot(index="Game 1", columns="Game 2", values=stat)
    df = df.reindex(order)
    df = df[order]
    fig, ax = plt.subplots()
    sns.heatmap(df, annot=True, cmap="Greens", ax=ax)
    ax.set(xlabel="Game", ylabel="Game")
    ax.set(title=f"{feature} {stat} Between Games")
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
    ax.set(title=f"{feature} {stat} Between Games")
    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/distinguish_bp_{feature}.png", bbox_inches="tight")


def main(feature_name, data_type):
    features_data_path = get_data_path(data_type, "features")
    save_loc = get_data_path(data_type, "images")
    df = pd.read_csv(f"{features_data_path}/all.csv")
    df = df[df["game"] != "Unknown"]
    
    if feature_name in df.columns:
        samples = df[[feature_name, "game"]].values
        get_difference = get_difference_feature
        stat = "Difference"
    else:
        samples = get_functions(features_data_path, feature_name, data_type)
        get_difference = get_difference_function
        stat = "KS Test p-value"

    data = []
    samples = samples[0:100]
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            feature_i, game_i = samples[i]
            feature_j, game_j = samples[j]
            diff = get_difference(feature_i, feature_j)
            data.append({"Game 1":game_i, "Game 2":game_j, stat:diff})
            if game_i != game_j:
                data.append({"Game 1":game_j, "Game 2":game_i, stat:diff})

    df_data = pd.DataFrame(data)
    plot_heatmap(save_loc, feature_name, df_data, stat)
    plot_barplot(save_loc, feature_name, df_data, stat)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the feature name and data type.")
