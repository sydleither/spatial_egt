import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

from common import plot_line, read_all


def euclidean_counts(df, exp_dir, dimension, fr="", transparent=False):
    df = df.loc[df["measure"] == "euclidean"]

    df["proportion"] = 0
    df.loc[df["pair"] == "SS", "proportion"] = df["count"] / (0.5*df["sensitive"]*(df["sensitive"]-1))
    df.loc[df["pair"] == "SR", "proportion"] = df["count"] / (0.5*df["total"]*(df["total"]-1))
    df.loc[df["pair"] == "RR", "proportion"] = df["count"] / (0.5*df["resistant"]*(df["resistant"]-1))

    colors = ["sienna", "hotpink", "limegreen", "royalblue"]
    for pair in df["pair"].unique():
        df_pair = df.loc[df["pair"] == pair]
        for time in df_pair["time"].unique():
            df_time = df_pair.loc[df_pair["time"] == time]
            fig, ax = plt.subplots()
            for i,condition in enumerate(df_time["condition"].unique()):
                df_cond = df_time.loc[df["condition"] == condition]
                plot_line(ax, df_cond, "distance", "proportion", colors[i], condition)
            ax.set_xlabel("Euclidean Distance Between Pairs of Cells")
            ax.set_ylabel("Proportion")
            ax.set_title(f"{dimension} {pair} pairs at time {time}")
            ax.legend()
            fig.tight_layout()
            if transparent:
                fig.patch.set_alpha(0.0)
            fig.savefig(f"output/{exp_dir}/euclidean_{dimension}pair_correlation_{pair}_{time}{fr}.png")


def pair_correlation_functions(df, exp_dir, dimension, conditions=[], fr="", transparent=False):
    df = df.loc[df["measure"] != "euclidean"]
    if len(conditions) == 0:
        conditions = df["condition"].unique()

    light_colors = ["sandybrown", "lightpink", "lightgreen", "skyblue"]
    colors = ["sienna", "hotpink", "limegreen", "royalblue"]
    dark_colors = ["saddlebrown", "deeppink", "darkgreen", "mediumblue"]
    for pair in df["pair"].unique():
        df_pair = df.loc[df["pair"] == pair]
        for time in df_pair["time"].unique():
            df_time = df_pair.loc[df_pair["time"] == time]
            fig, ax = plt.subplots()
            for i in range(len(conditions)):
                df_cond = df_time.loc[df["condition"] == conditions[i]]
                plot_line(ax, df_cond.loc[df["measure"] == "x"], "distance", "P", colors[i], conditions[i])
                plot_line(ax, df_cond.loc[df["measure"] == "y"], "distance", "P", light_colors[i], None)
                if dimension == "3D":
                    plot_line(ax, df_cond.loc[df["measure"] == "z"], "distance", "P", dark_colors[i], None)
            ax.set_xlabel("Distance Between Pairs of Cells")
            ax.set_ylabel("Pair-Correlation Function Signal")
            ax.set_title(f"{dimension} {pair} pairs at time {time}")
            ax.legend()
            fig.tight_layout()
            if transparent:
                fig.patch.set_alpha(0.0)
            fig.savefig(f"output/{exp_dir}/{dimension}pair_correlation_{pair}_{time}{fr}.png")


def main(exp_dir, dimension):
    df_pc = read_all(exp_dir, "pairCorrelations", dimension)
    df_pop = read_all(exp_dir, "populations", dimension)
    df = df_pop.merge(df_pc, on=["model", "time", "rep", "condition", "dimension"])
    df["total"] = df["sensitive"] + df["resistant"]

    #calculate pair-correlation function signal (Binder & Simpson, 2013)
    grid_area = 15625
    grid_len = 125 if dimension == "2D" else 25
    #probability of selecting an agent
    df["p"] = 0
    df.loc[df["pair"] == "SS", "p"] = df["sensitive"]/grid_area
    df.loc[df["pair"] == "SR", "p"] = df["total"]/grid_area
    df.loc[df["pair"] == "RR", "p"] = df["resistant"]/grid_area
    #probability of selecting a second agent
    df["p_hat"] = 0
    df.loc[df["pair"] == "SS", "p_hat"] = (df["sensitive"] - 1) / (grid_area - 1)
    df.loc[df["pair"] == "SR", "p_hat"] = (2*df["resistant"]*df["sensitive"]) / ((df["total"])*(df["total"]-1))
    df.loc[df["pair"] == "RR", "p_hat"] = (df["resistant"] - 1) / (grid_area - 1)
    #normalization term
    multiplier = grid_len**2 if dimension == "2D" else grid_len**4
    df["C"] = multiplier*(grid_len-df["distance"])*df["p"]*df["p_hat"]
    #pair correlation function
    df["P"] = df["count"] / df["C"]
    df.loc[df["P"] < 0, "P"] = 0

    if exp_dir == "gamespc":
        df["fr"] = df["condition"].str[-3:]
        df["condition"] = df["condition"].str[0:-3]
        for fr in df["fr"].unique():
            df_fr = df.loc[df["fr"] == fr]
            pair_correlation_functions(df_fr, exp_dir, dimension, fr="_"+fr)
            euclidean_counts(df_fr, exp_dir, dimension, fr="_"+fr)
    else:
        pair_correlation_functions(df, exp_dir, dimension, conditions=["bistability_75", "coexistence_75", "resistant_max", "sensitive_min"])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide an experiment directory and dimension.")