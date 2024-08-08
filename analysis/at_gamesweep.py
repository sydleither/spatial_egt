from itertools import product
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import read_specific, plot_line


COLORS = ["sienna", "limegreen", "royalblue", "darkviolet", "darkgray", 
          "indianred", "deeppink", "brown", "olive", "cyan"]


def plot_fs_by_r(df, exp_dir, exp_name, dimension, model, times):
    df = df.loc[df["model"] == model]
    fig, ax = plt.subplots()
    for i,time in enumerate(times):
        df_t = df.loc[df["time"] == time]
        df_t = df_t.sort_values("radius")
        if len(df_t["rep"].unique()) > 1:
            plot_line(ax, df_t, "radius", "fs", COLORS[i], time)
        else:
            ax.scatter(df_t["radius"], df_t["fs"], color=COLORS[i], label=time)
    ax.set(xlabel="Radius", ylabel="Fraction Sensitive", title=f"{exp_name} {model} {dimension}")
    fig.legend(title="Tick")
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/{exp_name}/fsr_{model}{dimension}.png")


def plot_gr_by_fs(df, exp_dir, exp_name, dimension, model):
    df = df.loc[df["model"] == model]
    radii = df["radius"].unique()
    fig, ax = plt.subplots(1, len(radii), figsize=(30, 3))
    for j,radius in enumerate(radii):
        for rep in df["rep"].unique():
            df_tr = df.loc[(df["radius"] == radius) & (df["rep"] == rep)]
            ax[j].scatter(df_tr["fs"], df_tr["resistant"], color=COLORS[rep])
            ax[j].set(title=radius)
    fig.supxlabel("Fraction Sensitive")
    fig.supylabel("Growth Rate of Resistant")
    fig.suptitle(f"{exp_name} {model} {dimension}")
    fig.tight_layout(rect=[0.015, 0.03, 1, 0.95])
    fig.savefig(f"output/{exp_dir}/{exp_name}/grfs_{model}{dimension}.png")


def plot_r_by_gr(df, exp_dir, exp_name, dimension, model):
    df = df.loc[df["model"] == model]
    fig, ax = plt.subplots()
    sns.violinplot(data=df, x="radius", y="resistant", hue="rep", ax=ax)
    ax.set(xlabel="Radius", ylabel="Growth Rate of Resistant", title=f"{exp_name} {model} {dimension}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"output/{exp_dir}/{exp_name}/rgr_{model}{dimension}.png")


def main(exp_dir, exp_name, dimension):
    df_fs = read_specific(exp_dir, exp_name, dimension, "fs")
    df_pop = read_specific(exp_dir, exp_name, dimension, "populations")
    df_pop = df_pop.reset_index()
    df_gr = df_pop.drop(["index", "time", "rep", "model"], axis=1)
    df_gr = df_gr.rolling(window=2).apply(lambda x: (x.values[1]-x.values[0])/x.values[0] if x.values[0] != 0 else x.values[1])
    df_gr = df_gr.join(df_pop[["time", "rep", "model"]])
    df = df_fs.merge(df_gr, on=["model", "time", "rep"])

    for model in df["model"].unique():
        plot_fs_by_r(df_fs, exp_dir, exp_name, dimension, model, [0, 1000, 5000, 10000])
        plot_gr_by_fs(df, exp_dir, exp_name, dimension, model)
        plot_r_by_gr(df, exp_dir, exp_name, dimension, model)


def theoretical_main(exp_dir, exp_name, dimension):
    config = json.load(open(f"output/{exp_dir}/{exp_name}/{exp_name}.json"))
    a = config["A"]
    b = config["B"]
    c = config["C"]
    d = config["D"]

    radius_areas = {1:4, 2:12, 3:28, 4:48, 5:80}
    df_dict = {"model":[], "rep":[], "time":[], "radius":[], "fs":[], "resistant":[], "sensitive":[]}
    for radius, radius_area in radius_areas.items():
        types_in_radius = [x for x in product(range(radius_area+1), repeat=3) if sum(x) == radius_area]
        for num_empty, num_sensitive, num_resistant in types_in_radius:
            total_cells = num_sensitive+num_resistant
            if total_cells == 0:
                continue
            df_dict["model"].append("theoretical")
            df_dict["rep"].append(0)
            df_dict["time"].append(0)
            df_dict["radius"].append(radius)
            df_dict["fs"].append(num_sensitive/total_cells)
            df_dict["resistant"].append(((num_sensitive*c) + (num_resistant*d))/total_cells)
            df_dict["sensitive"].append(((num_sensitive*a) + (num_resistant*b))/total_cells)
    df = pd.DataFrame(df_dict)
    df["resistant"] = df["resistant"].round(3)
    df["sensitive"] = df["sensitive"].round(3)
    df = df.drop_duplicates()

    plot_fs_by_r(df, exp_dir, exp_name, dimension, "theoretical", [0])
    plot_gr_by_fs(df, exp_dir, exp_name, dimension, "theoretical")
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