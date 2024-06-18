import sys
import warnings
warnings.filterwarnings("ignore")

from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd

from common import plot_line, read_dim


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


def pair_correlation_functions(df, grid_len, grid_area, exp_dir, exp_name, dimension, transparent=False):
    df = df.loc[df["measure"] != "euclidean"]

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

    colors = cm.get_cmap("Set3").colors
    for pair in df["pair"].unique():
        df_pair = df.loc[df["pair"] == pair]
        for time in [0, 500]:
            df_time = df_pair.loc[df_pair["time"] == time]
            fig, ax = plt.subplots()
            plot_line(ax, df_time.loc[df["measure"] == "x"], "distance", "P", colors[0], "x")
            plot_line(ax, df_time.loc[df["measure"] == "y"], "distance", "P", colors[1], "y")
            ax.set_xlabel("Distance Between Pairs of Cells")
            ax.set_ylabel("Normalized Count")
            ax.set_title(f"Pair Correlation for {dimension} {exp_name} {pair} pairs at time {time}")
            fig.tight_layout()
            if transparent:
                fig.patch.set_alpha(0.0)
            fig.savefig(f"output/{exp_dir}/{exp_name}/{dimension}pair_correlation_{pair}_{time}.png")



def main(exp_dir, exp_name, dimension):
    df_pc = read_dim(exp_dir, exp_name, dimension, "pairCorrelations")
    df_pop = read_dim(exp_dir, exp_name, dimension, "populations")
    df = df_pop.merge(df_pc, on=["model", "time", "rep"])
    df["total"] = df["sensitive"] + df["resistant"]

    df["proportion"] = 0
    df.loc[df["pair"] == "SS", "proportion"] = df["count"] / (0.5*df["sensitive"]*(df["sensitive"]-1))
    df.loc[df["pair"] == "SR", "proportion"] = df["count"] / (0.5*df["total"]*(df["total"]-1))
    df.loc[df["pair"] == "RR", "proportion"] = df["count"] / (0.5*df["resistant"]*(df["resistant"]-1))

    if len(df["model"].unique()) > 1:
        print("Please input data with only one model.")
        exit()

    pair_correlation_functions(df, 125 if dimension == "2D" else 25, 15625, exp_dir, exp_name, dimension, transparent=False)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide an experiment directory, name, and dimension.")