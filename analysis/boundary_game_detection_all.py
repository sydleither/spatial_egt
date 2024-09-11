import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from boundary_game_detection import get_possible_fs
from common import get_colors, read_all


def fs_dist(df, exp_dir, dimension, model, radius, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time) & (df["radius"] == radius)]
    if radius > 1:
        df_fs = pd.MultiIndex.from_product([
                df["condition"].unique(),
                df["rep"].unique(),
                get_possible_fs(radius)],
                names=["condition", "rep", "fs"])
        df = df.set_index(["condition", "rep", "fs"]).reindex(df_fs, fill_value=0).reset_index()
    df["fs"] = df["fs"].astype(str)
    df = df[["fs", "normalized_total", "rep", "condition"]]

    conditions = sorted(df["condition"].unique())
    if len(conditions) > 10:
        palette = ["lightgreen", "limegreen", "darkgreen", "sandybrown", "chocolate", "saddlebrown",
                   "cyan", "darkturquoise", "cadetblue", "orchid", "mediumorchid", "darkorchid"]
    else:
        palette = get_colors()

    ylims = {1:0.6, 2:0.2, 3:0.1}

    fig, ax = plt.subplots(figsize=(8,6))
    for c,condition in enumerate(conditions):
        df_c = df.loc[df["condition"] == condition]
        df_c = df_c.groupby(["condition", "fs"]).mean().reset_index()
        ax.bar(x=df_c["fs"], height=df_c["normalized_total"], color=palette[c], alpha=0.5, width=1, label=condition)
    if radius > 1:
        ax.xaxis.set_major_locator(ticker.LinearLocator(10))
    ax.set(ylim=(0, ylims[radius]),
           xlabel=f"Fraction Sensitive in Radius {radius} from Resistant", 
           ylabel="Proportion of Boundary Resistant", 
           title=f"{exp_dir} {model} {dimension} at tick {time}")
    ax.legend()
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/combofs_{model}{dimension}_r{radius}t{time}.png", bbox_inches="tight")
    plt.close()


def plot_average_fs(df, exp_dir, dimension, model, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time) & (df["radius"] <= 5)]
    df = df[["condition", "rep", "radius", "average_fs"]].drop_duplicates()

    conditions = sorted(df["condition"].unique())
    if len(conditions) > 10:
        palette = ["lightgreen", "limegreen", "darkgreen", "sandybrown", "chocolate", "saddlebrown",
                   "cyan", "darkturquoise", "cadetblue", "orchid", "mediumorchid", "darkorchid"]
    else:
        palette = get_colors()

    fig, ax = plt.subplots(figsize=(10,8))
    sns.lineplot(df, x="radius", y="average_fs", hue="condition", hue_order=conditions, errorbar="ci", palette=palette, ax=ax)
    ax.set(xlabel="Neighborhood Radius", 
           ylabel="Average Fraction Sensitive", 
           title=f"{exp_dir} {model} {dimension} at tick {time}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"output/{exp_dir}/avgfs_{model}{dimension}_t{time}.png")
    plt.close()


def main(exp_dir, dimension):
    df_key = ["model", "condition", "time", "radius", "rep"]
    df = read_all(exp_dir, "fs", dimension)
    df = df.loc[(df["fs"] < 1) & (df["fs"] > 0)]
    df["weighted_fs"] = df["fs"]*df["total"]
    df_grp = df[df_key+["total", "weighted_fs"]].groupby(df_key).sum().reset_index()
    df_grp = df_grp.rename(columns={"total":"total_boundary", "weighted_fs":"weighted_fs_sum"})
    df = df.merge(df_grp, on=df_key)
    df["average_fs"] = df["weighted_fs_sum"]/df["total_boundary"]
    df["normalized_total"] = df["total"] / df["total_boundary"]

    times = list(df["time"].unique())
    times.remove(0)
    radii = [1, 2, 3]
    for time in times:
        plot_average_fs(df, exp_dir, dimension, "nodrug", time)
        for radius in radii:
            fs_dist(df, exp_dir, dimension, "nodrug", radius, time)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide an experiment directory and dimension.")