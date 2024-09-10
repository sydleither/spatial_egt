from itertools import product
import json
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from common import read_specific


COLORS = ["#509154", "#A9561E", "#77BCFD", "#B791D4", "#EEDD5D",
          "#738696", "#24BCA8", "#D34A4F", "#8D81FE", "#FDA949"]


def boundary_r(df, exp_dir, exp_name, dimension, model, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time)]
    df = df[["rep", "radius", "total_boundary", "resistant"]].drop_duplicates()
    df["s_in_neighborhood"] = df["total_boundary"] / df["resistant"]
    
    fig, ax = plt.subplots()
    sns.barplot(df, x="radius", y="s_in_neighborhood", errorbar="ci", color=COLORS[0], ax=ax)
    ax.set(ylim=(0, 1),
           xlabel="Neighborhood Radius", 
           ylabel="Proportion of Resistant Cells with a\nSensitive Cell in Their Neighborhood", 
           title=f"{exp_name} {model} {dimension} at tick {time}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"output/{exp_dir}/{exp_name}/boundary_{model}{dimension}_t{time}.png")
    plt.close()


def get_possible_fs(radius):
    radius_areas = {1:4, 2:12, 3:28, 4:48, 5:80}
    radius_area = radius_areas[radius]
    fs = set()
    types_in_radius = [x for x in product(range(radius_area+1), repeat=3) if sum(x) == radius_area]
    for num_empty, num_sensitive, num_resistant in types_in_radius:
        total_cells = num_sensitive+num_resistant
        if total_cells == 0:
            continue
        fs.add(round(num_sensitive/total_cells, 2))
    return fs


def fs_dist(df, exp_dir, exp_name, dimension, model, radius, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time) & (df["radius"] == radius)]
    if radius > 1:
        df_fs = pd.MultiIndex.from_product([
                df["rep"].unique(),
                get_possible_fs(radius)],
                names=["rep", "fs"])
        df = df.set_index(["rep", "fs"]).reindex(df_fs, fill_value=0).reset_index()
        df = df[["fs", "normalized_total", "rep"]]

    fig, ax = plt.subplots()
    sns.barplot(df, x="fs", y="normalized_total", errorbar="ci", color=COLORS[0], ax=ax)
    if radius > 1:
        ax.xaxis.set_major_locator(ticker.LinearLocator(10))
    ax.set(ylim=(0, 0.5/radius),
           xlabel=f"Fraction Sensitive In Radius {radius} from Resistant", 
           ylabel="Proportion of Boundary Resistant", 
           title=f"{exp_name} {model} {dimension} at tick {time}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"output/{exp_dir}/{exp_name}/fs_{model}{dimension}_r{radius}t{time}.png")
    plt.close()


def main(exp_dir, exp_name, dimension):
    df_key = ["model", "time", "radius", "rep"]
    df = read_specific(exp_dir, exp_name, dimension, "fs")
    df = df.loc[(df["fs"] < 1) & (df["fs"] > 0)]
    df_grp = df[df_key+["total"]].groupby(df_key).sum().reset_index()
    df_grp = df_grp.rename(columns={"total":"total_boundary"})
    df = df.merge(df_grp, on=df_key)
    df["normalized_total"] = df["total"] / df["total_boundary"]

    fs_dist(df, exp_dir, exp_name, dimension, "nodrug", 3, 500)

    df_pop = read_specific(exp_dir, exp_name, dimension, "populations")
    df = df.merge(df_pop, on=["model", "rep", "time"])
    boundary_r(df, exp_dir, exp_name, dimension, "nodrug", 500)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide an experiment directory, experiment name, and dimension.")