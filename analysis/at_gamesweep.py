from itertools import product
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import read_specific


COLORS = ["#509154", "#A9561E", "#77BCFD", "#B791D4", "#EEDD5D", "#738696", "#24BCA8", "#D34A4F", "#8D81FE", "#FDA949"]


def plot_r_by_fs(df, exp_dir, exp_name, dimension, model):
    df = df.loc[df["model"] == model]
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="radius", y="fs", orient="x", ax=ax, hue="time", palette=COLORS)
    #sns.lineplot(data=df, x="radius", y="fs", orient="x", ax=ax, hue="time", palette=COLORS)
    ax.set(xlabel="Radius", ylabel="Fraction Sensitive", title=f"{exp_name} {model} {dimension}")
    ax.set(ylim=(0, 1))
    fig.patch.set_alpha(0.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"output/{exp_dir}/{exp_name}/rfs_{model}{dimension}.png")


def plot_fs_by_gr(df, exp_dir, exp_name, dimension, model, limit_radii=False):
    df = df.loc[df["model"] == model]
    if limit_radii:
        radii = [1,2,3,4,5]
        fig_width = 15
    else:
        radii = sorted(df["radius"].unique())
        fig_width = 30
    fig, ax = plt.subplots(1, len(radii), figsize=(fig_width, 3))
    for i,radius in enumerate(radii):
        df_r = df.loc[df["radius"] == radius]
        #sns.lineplot(data=df_r, x="fs", y="gr", orient="x", ax=ax[j], color=COLORS[0])
        for j,time in enumerate(df_r["time"].unique()):
            df_rt = df_r.loc[df_r["time"] == time]
            ax[i].scatter(x=df_rt["fs"], y=df_rt["gr"], color=COLORS[j], alpha=0.66)
        ax[i].set(title=radius, xlim=(0, 1), ylim=(0, 0.1))
    fig.supxlabel("Fraction Sensitive")
    fig.supylabel("Resistant Growth Rate")
    fig.suptitle(f"{exp_name} {model} {dimension}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout(rect=[0.015, 0.03, 1, 0.95])
    fig.savefig(f"output/{exp_dir}/{exp_name}/fsgr_{model}{dimension}.png")


def plot_r_by_gr(df, exp_dir, exp_name, dimension, model):
    df = df.loc[df["model"] == model]
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(data=df, x="radius", y="delta_gr", notch=True, ax=ax, color=COLORS[0])
    ax.axhline(0, linestyle="dashed", color="black", linewidth=0.5)
    ax.set(xlabel="Radius", ylabel="Delta Resistant Growth Rate", title=f"{exp_name} {model} {dimension}")
    ax.set(ylim=(-1, 1))
    fig.patch.set_alpha(0.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"output/{exp_dir}/{exp_name}/rgr_{model}{dimension}.png")


def main(exp_dir, exp_name, dimension):
    config = json.load(open(f"output/{exp_dir}/{exp_name}/{exp_name}.json"))
    df_fs = read_specific(exp_dir, exp_name, dimension, "fs")
    df_fs = df_fs.loc[(df_fs["time"] > 0) & (df_fs["time"] <= 1500)]
    df_fs["gr"] = (df_fs["reproduced"] / df_fs["total"])
    df = df_fs[["model", "time", "radius", "fs", "reproduced", "total", "gr"]].groupby(["model", "time", "radius", "fs"]).mean().reset_index()
    df["delta_gr"] = (df["gr"] - config["D"]) / config["D"]

    for model in df["model"].unique():
        plot_r_by_fs(df, exp_dir, exp_name, dimension, model)
        plot_fs_by_gr(df, exp_dir, exp_name, dimension, model, limit_radii=True)
        plot_r_by_gr(df, exp_dir, exp_name, dimension, model)


def theoretical_main(exp_dir, exp_name, dimension):
    config = json.load(open(f"output/{exp_dir}/{exp_name}/{exp_name}.json"))
    a = config["A"]
    b = config["B"]
    c = config["C"]
    d = config["D"]

    radius_areas = {1:4, 2:12, 3:28, 4:48, 5:80}
    df_dict = {"radius":[], "fs":[], "gr":[], "sensitive":[]}
    for radius, radius_area in radius_areas.items():
        types_in_radius = [x for x in product(range(radius_area+1), repeat=3) if sum(x) == radius_area]
        for num_empty, num_sensitive, num_resistant in types_in_radius:
            total_cells = num_sensitive+num_resistant
            if total_cells == 0:
                continue
            df_dict["radius"].append(radius)
            df_dict["fs"].append(num_sensitive/total_cells)
            df_dict["gr"].append(((num_sensitive*c) + (num_resistant*d))/total_cells)
            df_dict["sensitive"].append(((num_sensitive*a) + (num_resistant*b))/total_cells)
    df = pd.DataFrame(df_dict)
    df["model"] = "theoretical"
    df["rep"] = 0
    df["time"] = 0
    df["total"] = 1
    df["gr"] = df["gr"].round(3)
    df["sensitive"] = df["sensitive"].round(3)
    df["delta_gr"] = (df["gr"] - d) / d
    df = df.drop_duplicates()

    plot_r_by_fs(df, exp_dir, exp_name, dimension, "theoretical")
    plot_fs_by_gr(df, exp_dir, exp_name, dimension, "theoretical", limit_radii=True)
    plot_r_by_gr(df, exp_dir, exp_name, dimension, "theoretical")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        analysis_type = sys.argv[4]
        if analysis_type == "theoretical":
            theoretical_main(sys.argv[1], sys.argv[2], sys.argv[3])
        else:
            main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide an experiment directory, experiment name, dimension, and analysis type.")