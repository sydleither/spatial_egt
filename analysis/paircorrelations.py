import sys
import warnings
warnings.filterwarnings("ignore")

from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd

from common import plot_line, read_all


def euclidean_histograms(df, exp_dir, exp_name, dimension, transparent=False):
    df = df.loc[df["measure"] == "euclidean"]
    colors = cm.get_cmap("Set3").colors
    for pair in df["pair"].unique():
        df_pair = df.loc[df["pair"] == pair]
        for time in df_pair["time"].unique():
            df_time = df_pair.loc[df_pair["time"] == time]
            fig, ax = plt.subplots()
            for rep in df_time["rep"].unique():
                df_rep = df_time.loc[df_time["rep"] == rep]
                ax.bar(df_rep["distance"], df_rep["count"], alpha=0.6, color=colors[rep])
            ax.set_xlabel("Distance Between Pairs of Cells")
            ax.set_ylabel("Count of Pairs")
            ax.set_title(f"Pair Correlation for {dimension} {exp_name} {pair} pairs at time {time}")
            fig.tight_layout()
            if transparent:
                fig.patch.set_alpha(0.0)
            fig.savefig(f"output/{exp_dir}/{exp_name}/{dimension}pair_correlation_{pair}_{time}.png")


def pair_correlation_functions(df, exp_dir, dimension, fr="", transparent=False):
    df = df.loc[df["measure"] != "euclidean"]

    time_end = df["time"].max()
    light_colors = ["sandybrown", "lightpink", "lightgreen"]
    dark_colors = ["sienna", "hotpink", "limegreen"]
    for pair in df["pair"].unique():
        df_pair = df.loc[df["pair"] == pair]
        for time in [0, time_end]:
            df_time = df_pair.loc[df_pair["time"] == time]
            fig, ax = plt.subplots()
            for i,condition in enumerate(df_time["condition"].unique()):
                df_cond = df_time.loc[df["condition"] == condition]
                plot_line(ax, df_cond.loc[df["measure"] == "x"], "distance", "P", dark_colors[i], condition+"_x")
                plot_line(ax, df_cond.loc[df["measure"] == "y"], "distance", "P", light_colors[i], condition+"_y")
            ax.set_xlabel("Distance Between Pairs of Cells")
            ax.set_ylabel("Pair-Correlation Function Signals")
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

    #calculate pair-correlation function signal
    grid_area = 15625
    grid_len = 125 if dimension == "2D" else 25
    #probability of selecting an agent
    df["p"] = 0
    df.loc[df["pair"] == "SS", "p"] = df["sensitive"]/grid_area
    df.loc[df["pair"] == "SR", "p"] = df["total"]/grid_area
    df.loc[df["pair"] == "RR", "p"] = df["resistant"]/grid_area
    #probability of selecting a second agent
    ga_inv = 1/grid_area
    df["p_hat"] = (df["p"]-(ga_inv)) / (1-ga_inv)
    #normalization term
    df["C"] = grid_area*(grid_len-df["distance"])*df["p"]*df["p_hat"]
    #pair correlation function
    df["P"] = df["count"] / df["C"]

    if exp_dir == "gamespc":
        df["fr"] = df["condition"].str[-3:]
        df["condition"] = df["condition"].str[0:-3]
        for fr in df["fr"].unique():
            df_fr = df.loc[df["fr"] == fr]
            pair_correlation_functions(df_fr, exp_dir, dimension, fr="_"+fr)
    else:
        pair_correlation_functions(df, 125 if dimension == "2D" else 25, 15625, exp_dir, dimension)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide an experiment directory and dimension.")