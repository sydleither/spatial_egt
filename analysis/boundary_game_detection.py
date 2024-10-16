from itertools import product
import json
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from common import get_colors, process_fi, read_specific


COLORS = get_colors()


def boundary_i(df, fi, exp_dir, exp_name, dimension, model, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time) & (df["radius"] <= 5)]
    focus_cell = "resistant" if fi == "fs" else "sensitive"
    neighbor_cell = "resistant" if fi == "fr" else "sensitive"
    df = df[["rep", "radius", "total_boundary", focus_cell]].drop_duplicates()
    df["i_in_neighborhood"] = df["total_boundary"] / df[focus_cell]
    
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(df, x="radius", y="i_in_neighborhood", errorbar="ci", color=COLORS[0], ax=ax)
    ax.set(ylim=(0, 1),
           xlabel="Neighborhood Radius", 
           ylabel=f"Proportion of {focus_cell} Cells with a\n{neighbor_cell} Cell in Their Neighborhood", 
           title=f"{exp_dir} {model} {dimension} at tick {time}")
    ax.legend()
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/{exp_name}/boundary_{focus_cell}_{model}{dimension}_t{time}.png", bbox_inches="tight")
    plt.close()


def get_possible_fi(radius, fi):
    radius_areas = {1:4, 2:12, 3:28, 4:48, 5:80}
    radius_area = radius_areas[radius]
    fi_set = set()
    types_in_radius = [x for x in product(range(radius_area+1), repeat=3) if sum(x) == radius_area]
    for num_empty, num_sensitive, num_resistant in types_in_radius:
        total_cells = num_sensitive+num_resistant
        if total_cells == 0:
            continue
        if fi == "fs":
            fi_set.add(round(num_sensitive/total_cells, 2))
        else:
            fi_set.add(round(num_resistant/total_cells, 2))
    return fi_set


def fi_dist(df, fi, exp_dir, exp_name, dimension, model, radius, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time) & (df["radius"] == radius)]
    if radius > 1 and model != "theoretical":
        df_fr = pd.MultiIndex.from_product([
                df["rep"].unique(),
                get_possible_fi(radius, fi)],
                names=["rep", fi])
        df = df.set_index(["rep", fi]).reindex(df_fr, fill_value=0).reset_index()
        df = df[[fi, "normalized_total", "rep"]]
    focus_cell = "Resistant" if fi == "fs" else "Sensitive"
    neighbor_cell = "Resistant" if fi == "fr" else "Sensitive"

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(df, x=fi, y="normalized_total", errorbar="ci", color=COLORS[0], ax=ax)
    if radius > 1:
        ax.xaxis.set_major_locator(ticker.LinearLocator(10))
    if model == "theoretical":
        ax.set(ylim=(0, 0.05),
               xlabel=f"Fraction {neighbor_cell} in Radius {radius} from {focus_cell}", 
               ylabel=f"Proportion of Boundary {focus_cell}", 
               title=f"{exp_name} {model} {dimension}")
    else:
        ax.set(ylim=(0, 0.5/radius),
               xlabel=f"Fraction {neighbor_cell} in Radius {radius} from {focus_cell}", 
               ylabel=f"Proportion of Boundary {focus_cell}", 
               title=f"{exp_name} {model} {dimension} at tick {time}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/{exp_name}/{fi}_{model}{dimension}_r{radius}t{time}.png", bbox_inches="tight")
    plt.close()


def plot_average_fi(df, fi, exp_name, exp_dir, dimension, model, time):
    df = df.loc[(df["model"] == model) & (df["time"] == time) & (df["radius"] <= 5)]
    df = df[["rep", "radius", f"average_{fi}"]].drop_duplicates()
    cell = "Sensitive" if fi == "fs" else "Resistant"

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(df, x="radius", y=f"average_{fi}", errorbar="ci", color=COLORS[0], ax=ax)
    ax.set(ylim=(0, 0.5),
           xlabel="Neighborhood Radius", 
           ylabel=f"Average Fraction {cell}",
           title=f"{exp_name} {model} {dimension} at tick {time}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/{exp_name}/avg{fi}_{model}{dimension}_t{time}.png", bbox_inches="tight")
    plt.close()


def main(exp_dir, exp_name, dimension, fi):
    df_key = ["model", "time", "radius", "rep"]
    df = process_fi(read_specific(exp_dir, exp_name, dimension, fi), fi, df_key)

    times = list(df["time"].unique())
    if 0 in times:
        times.remove(0)
    radii = [1, 2, 3]
    for time in times:
        plot_average_fi(df, fi, exp_name, exp_dir, dimension, "nodrug", time)
        for radius in radii:
            fi_dist(df, fi, exp_dir, exp_name, dimension, "nodrug", radius, time)

    df_pop = read_specific(exp_dir, exp_name, dimension, "populations")
    df = df.merge(df_pop, on=["model", "rep", "time"])
    for time in times:
        boundary_i(df, fi, exp_dir, exp_name, dimension, "nodrug", time)


def fr_hist(df, fi, exp_dir, exp_name, dimension, radius):
    df = df.loc[(df["radius"] == radius)]
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=fi, stat="proportion", color=COLORS[0], binwidth=0.01, ax=ax)
    ax.xaxis.set_major_locator(ticker.LinearLocator(10))
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/{exp_name}/{fi}hist{dimension}.png", bbox_inches="tight")


def theoretical_main(exp_dir, exp_name, dimension, fi):
    config = json.load(open(f"output/{exp_dir}/{exp_name}/{exp_name}.json"))
    a = config["A"]
    b = config["B"]
    c = config["C"]
    d = config["D"]

    radius_areas = {1:4, 2:12, 3:28, 4:48, 5:80}
    df_dict = {"radius":[], "fr":[], "fs":[], "resistant_gr":[], "sensitive_gr":[]}
    for radius, radius_area in radius_areas.items():
        types_in_radius = [x for x in product(range(radius_area+1), repeat=3) if sum(x) == radius_area]
        for num_empty, num_sensitive, num_resistant in types_in_radius:
            total_cells = num_sensitive+num_resistant
            if total_cells == 0:
                continue
            df_dict["radius"].append(radius)
            df_dict["fr"].append(num_resistant/total_cells)
            df_dict["fs"].append(num_sensitive/total_cells)
            df_dict["resistant_gr"].append(((num_sensitive*c) + (num_resistant*d))/total_cells)
            df_dict["sensitive_gr"].append(((num_sensitive*a) + (num_resistant*b))/total_cells)
    df = pd.DataFrame(df_dict)
    df["model"] = "theoretical"
    df["rep"] = 0
    df["time"] = 0
    df["total"] = 1
    df["fr"] = df["fr"].round(2)
    df["fs"] = df["fs"].round(2)
    df["resistant_gr"] = df["resistant_gr"].round(3)
    df["sensitive_gr"] = df["sensitive_gr"].round(3)

    if fi == "fs":
        df["normalized_total"] = df["resistant_gr"]
    else:
        df["normalized_total"] = df["sensitive_gr"]
    for radius in [1, 2, 3]:
        fr_hist(df, fi, exp_dir, exp_name, dimension, radius)
        fi_dist(df, fi, exp_dir, exp_name, dimension, "theoretical", radius, 0)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 6:
        theoretical_main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Please provide an experiment directory, experiment name, \
              dimension, fr or fs, and optionally a flag to run theoretical analysis.")