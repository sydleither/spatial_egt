import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import read_specific, plot_line


def plot_fs_by_r(df, exp_dir, exp_name, dimension, model):
    df = df.loc[df["model"] == model]
    fig, ax = plt.subplots()
    colors = ["sienna", "limegreen", "royalblue", "darkviolet", "darkgray", "indianred"]
    for i,time in enumerate([0, 50, 100]):
        df_t = df.loc[df["time"] == time]
        df_t = df_t.sort_values("radius")
        plot_line(ax, df_t, "radius", "fs", colors[i], time)
    ax.set(xlabel="Radius", ylabel="Fraction Sensitive", title=f"{exp_name} {model} {dimension}")
    fig.legend(title="Tick")
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/{exp_name}/fsr_{model}{dimension}.png")


def plot_gr_by_fs(df, exp_dir, exp_name, dimension, model):
    df = df.loc[df["model"] == model]
    radii = df["radius"].unique()
    fig, ax = plt.subplots(1, len(radii), figsize=(30, 3))
    colors = ["sienna", "limegreen", "royalblue", "darkviolet", "darkgray", "indianred"]
    for j,radius in enumerate(radii):
        for rep in df["rep"].unique():
            df_tr = df.loc[(df["radius"] == radius) & (df["rep"] == rep)]
            ax[j].scatter(df_tr["fs"], df_tr["resistant"], color=colors[rep])
            ax[j].set(title=radius)
    fig.supxlabel("Fraction Sensitive")
    fig.supylabel("Growth Rate of Resistant")
    fig.suptitle(f"{exp_name} {model} {dimension}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
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
        plot_fs_by_r(df_fs, exp_dir, exp_name, dimension, model)
        plot_gr_by_fs(df, exp_dir, exp_name, dimension, model)
        plot_r_by_gr(df, exp_dir, exp_name, dimension, model)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide an experiment directory, experiment name, and dimension.")


# def main(file_name, c, d):
#     c = float(c)
#     d = float(d)
#     radius_areas = {1:4, 2:12, 3:28, 4:48, 5:80}

#     for radius in radius_areas:
#         radius_area = radius_areas[radius]
#         fs = [s/radius_area for s in range(radius_area+1)]
#         g = [((s*c)+((radius_area-s)*d))/radius_area for s in range(radius_area+1)]
#         print(fs)
#         print(g)
#         exit()