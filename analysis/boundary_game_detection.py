from itertools import product
import json
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from common import get_colors, read_specific


COLORS = get_colors()


def boundary_r(df, exp_dir, exp_name, dimension, model, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time) & (df["radius"] <= 5)]
    df = df[["rep", "radius", "total_boundary", "resistant"]].drop_duplicates()
    df["s_in_neighborhood"] = df["total_boundary"] / df["resistant"]
    
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(df, x="radius", y="s_in_neighborhood", errorbar="ci", color=COLORS[0], ax=ax)
    ax.set(ylim=(0, 1),
           xlabel="Neighborhood Radius", 
           ylabel="Proportion of Resistant Cells with a\nSensitive Cell in Their Neighborhood", 
           title=f"{exp_name} {model} {dimension} at tick {time}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/{exp_name}/boundary_{model}{dimension}_t{time}.png", bbox_inches="tight")
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
    if radius > 1 and model != "theoretical":
        df_fs = pd.MultiIndex.from_product([
                df["rep"].unique(),
                get_possible_fs(radius)],
                names=["rep", "fs"])
        df = df.set_index(["rep", "fs"]).reindex(df_fs, fill_value=0).reset_index()
        df = df[["fs", "normalized_total", "rep"]]

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(df, x="fs", y="normalized_total", errorbar="ci", color=COLORS[0], ax=ax)
    if radius > 1:
        ax.xaxis.set_major_locator(ticker.LinearLocator(10))
    if model == "theoretical":
        ax.set(ylim=(0, 0.05),
           xlabel=f"Fraction Sensitive in Radius {radius} from Resistant", 
           ylabel="Growth Rate", 
           title=f"{exp_name} {model} {dimension}")
    else:
        ax.set(ylim=(0, 0.5/radius),
            xlabel=f"Fraction Sensitive in Radius {radius} from Resistant", 
            ylabel="Proportion of Boundary Resistant", 
            title=f"{exp_name} {model} {dimension} at tick {time}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/{exp_name}/fs_{model}{dimension}_r{radius}t{time}.png", bbox_inches="tight")
    plt.close()


def plot_average_fs(df, exp_name, exp_dir, dimension, model, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time) & (df["radius"] <= 5)]
    df = df[["rep", "radius", "average_fs"]].drop_duplicates()

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(df, x="radius", y="average_fs", errorbar="ci", color=COLORS[0], ax=ax)
    ax.set(ylim=(0, 0.5),
           xlabel="Neighborhood Radius", 
           ylabel="Average Fraction Sensitive", 
           title=f"{exp_name} {model} {dimension} at tick {time}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/{exp_name}/avgfs_{model}{dimension}_t{time}.png", bbox_inches="tight")
    plt.close()


def main(exp_dir, exp_name, dimension):
    df_key = ["model", "time", "radius", "rep"]
    df = read_specific(exp_dir, exp_name, dimension, "fs")
    df = df.loc[df["fs"] > 0]
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
        plot_average_fs(df, exp_name, exp_dir, dimension, "nodrug", time)
        for radius in radii:
            fs_dist(df, exp_dir, exp_name, dimension, "nodrug", radius, time)

    df_pop = read_specific(exp_dir, exp_name, dimension, "populations")
    df = df.merge(df_pop, on=["model", "rep", "time"])
    for time in times:
        boundary_r(df, exp_dir, exp_name, dimension, "nodrug", time)


def theoretical_main(exp_dir, exp_name, dimension):
    config = json.load(open(f"output/{exp_dir}/{exp_name}/{exp_name}.json"))
    a = config["A"]
    b = config["B"]
    c = config["C"]
    d = config["D"]

    radius_areas = {1:4, 2:12, 3:28, 4:48, 5:80}
    df_dict = {"radius":[], "fs":[], "resistant_gr":[]}
    for radius, radius_area in radius_areas.items():
        types_in_radius = [x for x in product(range(radius_area+1), repeat=3) if sum(x) == radius_area]
        for num_empty, num_sensitive, num_resistant in types_in_radius:
            total_cells = num_sensitive+num_resistant
            if total_cells == 0:
                continue
            df_dict["radius"].append(radius)
            df_dict["fs"].append(num_sensitive/total_cells)
            df_dict["resistant_gr"].append(((num_sensitive*c) + (num_resistant*d))/total_cells)
    df = pd.DataFrame(df_dict)
    df["model"] = "theoretical"
    df["rep"] = 0
    df["time"] = 0
    df["total"] = 1
    df["fs"] = df["fs"].round(2)
    df["resistant_gr"] = df["resistant_gr"].round(3)

    df["normalized_total"] = df["resistant_gr"]
    for radius in [1, 2, 3]:
        fs_dist(df, exp_dir, exp_name, dimension, "theoretical", radius, 0)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        theoretical_main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide an experiment directory, experiment name, \
              dimension, and optionally a flag to run theoretical analysis.")