import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from boundary_game_detection import get_possible_fi
from common import get_colors, plot_line, process_fi, read_all


def get_conditions_palette(conditions):
    if len(conditions) > 10:
        palette = ["lightgreen", "limegreen", "darkgreen", "sandybrown", "chocolate", "saddlebrown",
                   "cyan", "darkturquoise", "cadetblue", "orchid", "mediumorchid", "darkorchid"]
    else:
        palette = get_colors()
    return palette


def fi_dist(df, fi, exp_dir, dimension, model, radius, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time) & (df["radius"] == radius)]
    if radius > 1:
        df_fi = pd.MultiIndex.from_product([
                df["condition"].unique(),
                df["rep"].unique(),
                get_possible_fi(radius, fi)],
                names=["condition", "rep", fi])
        df = df.set_index(["condition", "rep", fi]).reindex(df_fi, fill_value=0).reset_index()
    df[fi] = df[fi].astype(str)
    df = df[[fi, "normalized_total", "rep", "condition"]]
    focus_cell = "Resistant" if fi == "fs" else "Sensitive"
    neighbor_cell = "Resistant" if fi == "fr" else "Sensitive"

    ylims = {1:0.6, 2:0.2, 3:0.1}
    conditions = sorted(df["condition"].unique())
    palette = get_conditions_palette(conditions)

    fig, ax = plt.subplots(figsize=(8,6))
    for c,condition in enumerate(conditions):
        df_c = df.loc[df["condition"] == condition]
        df_c = df_c.groupby(["condition", fi]).mean().reset_index()
        ax.bar(x=df_c[fi], height=df_c["normalized_total"], color=palette[c], alpha=0.5, width=1, label=condition)
    if radius > 1:
        ax.xaxis.set_major_locator(ticker.LinearLocator(10))
    ax.set(ylim=(0, ylims[radius]),
           xlabel=f"Fraction {neighbor_cell} in Radius {radius} from {focus_cell}", 
           ylabel=f"Proportion of Boundary {focus_cell}", 
           title=f"{exp_dir} {model} {dimension} at tick {time}")
    ax.legend()
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/combo{fi}_{model}{dimension}_r{radius}t{time}.png", bbox_inches="tight")
    plt.close()


def boundary_i(df, fi, exp_dir, dimension, model, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time) & (df["radius"] <= 5)]
    focus_cell = "resistant" if fi == "fs" else "sensitive"
    neighbor_cell = "resistant" if fi == "fr" else "sensitive"
    df = df[["condition", "rep", "radius", "total_boundary", focus_cell]].drop_duplicates()
    df["i_in_neighborhood"] = df["total_boundary"] / df[focus_cell]

    conditions = sorted(df["condition"].unique())
    palette = get_conditions_palette(conditions)
    
    fig, ax = plt.subplots(figsize=(8,6))
    for c,condition in enumerate(conditions):
        df_cond = df.loc[df["condition"] == condition]
        plot_line(ax, df_cond, "radius", "i_in_neighborhood", palette[c], condition)
    ax.set(ylim=(0, 1),
           xlabel="Neighborhood Radius", 
           ylabel=f"Proportion of {focus_cell} Cells with a\n{neighbor_cell} Cell in Their Neighborhood", 
           title=f"{exp_dir} {model} {dimension} at tick {time}")
    ax.legend()
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/boundary_{focus_cell}_{model}{dimension}_t{time}.png", bbox_inches="tight")
    plt.close()


def plot_average_fi(df, fi, exp_dir, dimension, model, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time) & (df["radius"] <= 5)]
    df = df[["condition", "rep", "radius", f"average_{fi}"]].drop_duplicates()
    cell = "Sensitive" if fi == "fs" else "Resistant"

    conditions = sorted(df["condition"].unique())
    palette = get_conditions_palette(conditions)

    fig, ax = plt.subplots(figsize=(8,6))
    for c,condition in enumerate(conditions):
        df_cond = df.loc[df["condition"] == condition]
        plot_line(ax, df_cond, "radius", f"average_{fi}", palette[c], condition)
    ax.set(ylim=(0, 0.5),
           xlabel="Neighborhood Radius", 
           ylabel=f"Average Fraction {cell}", 
           title=f"{exp_dir} {model} {dimension} at tick {time}")
    ax.legend()
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/avg{fi}_{model}{dimension}_t{time}.png", bbox_inches="tight")
    plt.close()


def main(exp_dir, dimension, fi):
    df_key = ["model", "condition", "time", "radius", "rep"]
    df = process_fi(read_all(exp_dir, fi, dimension), fi, df_key)

    times = list(df["time"].unique())
    if 0 in times:
        times.remove(0)
    radii = [1, 2, 3]
    for time in times:
        plot_average_fi(df, fi, exp_dir, dimension, "nodrug", time)
        for radius in radii:
            fi_dist(df, fi, exp_dir, dimension, "nodrug", radius, time)

    df_pop = read_all(exp_dir, "populations", dimension)
    df = df.merge(df_pop, on=["model", "condition", "rep", "time"])
    for time in times:
        boundary_i(df, fi, exp_dir, dimension, "nodrug", time)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide an experiment directory, dimension, and fr or fs.")