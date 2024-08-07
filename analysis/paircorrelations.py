import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import plot_line, read_all


def pair_correlation_functions(df, exp_dir, dimension, freq_col="time", freq_ticks=[], conditions=[], transparent=False):
    df = df.loc[df["measure"] != "euclidean"]
    if freq_col == "confluence":
        df = groupby_confluence(df)
    if len(freq_ticks) == 0:
        freq_ticks = df[freq_col].unique()
    if len(conditions) == 0:
        conditions = sorted(df["condition"].unique())

    colors = ["lightgreen", "limegreen", "darkgreen", "sandybrown", "sienna", "saddlebrown",
              "cyan", "darkturquoise", "cadetblue", "orchid", "mediumorchid", "darkorchid"]
    #colors = ["sienna", "limegreen", "royalblue", "darkviolet", "darkgray", "indianred"]
    for pair in ["SR", "RS"]:
        df_pair = df.loc[df["pair"] == pair]
        for freq in freq_ticks:
            df_time = df_pair.loc[df_pair[freq_col] == freq]
            fig, ax = plt.subplots()
            for i in range(len(conditions)):
                df_cond = df_time.loc[df_time["condition"] == conditions[i]]
                if freq_col == "confluence":
                    df_cond = df_cond.loc[df_cond["time"] == df_cond["time"].min()]
                    print(conditions[i], freq, df_cond["time"].unique())
                if len(df_cond) == 0:
                    continue
                plot_line(ax, df_cond.loc[df_cond["measure"] == "annulus"], "radius", "pc", colors[i], conditions[i])
            ax.set_xlabel("Distance Between Pairs of Cells")
            ax.set_ylabel("Pair-Correlation Function Signal")
            ax.set_title(f"{dimension} {pair} pairs at {freq}")
            ax.legend()
            fig.tight_layout()
            if transparent:
                fig.patch.set_alpha(0.0)
            fig.savefig(f"output/{exp_dir}/{dimension}pair_correlation_{pair}_{freq}.png")


def groupby_confluence(df):
    df["confluence"] = np.round(df["total"], -3)
    df_grp = df.groupby(["model", "condition", "rep", "dimension", "radius", "pair", "measure", "confluence"]).first()
    df_grp = df_grp.reset_index()
    for cond in df_grp["condition"].unique():
        print(cond, list(df_grp.loc[df_grp["condition"] == cond]["confluence"].unique()))
    return df_grp


def bull_pc(df):
    df["pc"] = df["normalized_count"] / (df["sensitive"]*df["resistant"])
    return df


def main(exp_dir, dimension):
    df_pc = read_all(exp_dir, "pairCorrelations", dimension)
    df_pop = read_all(exp_dir, "populations", dimension)
    df = df_pop.merge(df_pc, on=["model", "time", "rep", "condition", "dimension"])
    df = df.loc[df["radius"] < 10]
    df["total"] = df["sensitive"] + df["resistant"]
    df = bull_pc(df)

    # pair_correlation_functions(df, exp_dir, dimension, freq_col="confluence", freq_ticks=[3000], transparent=True,
    #                             conditions=["sensitive_equal", "sensitive_agtb", "sensitive_bgta"])
    # pair_correlation_functions(df, exp_dir, dimension, freq_col="time", freq_ticks=[0, 1000, 2000], transparent=True,
    #                             conditions=["sensitive_equal", "sensitive_agtb", "sensitive_bgta"])
    # pair_correlation_functions(df, exp_dir, dimension, freq_col="time", freq_ticks=[0, 1000, 2000], transparent=True,
    #                             conditions=["coexistence_equal", "coexistence_bgtc", "coexistence_cgtb"])
    pair_correlation_functions(df, exp_dir, dimension, freq_col="time", freq_ticks=[0, 1000, 2000], transparent=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide an experiment directory and dimension.")