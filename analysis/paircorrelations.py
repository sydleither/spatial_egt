import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import plot_line, read_all


def pair_correlation_functions(df, exp_dir, dimension, freq_col="time", freq_ticks=[], conditions=[], transparent=False, zoom=False):
    if zoom:
        df = df.loc[df["radius"] <= 5]
    if freq_col == "confluence":
        df = groupby_confluence(df)
    if len(freq_ticks) == 0:
        freq_ticks = df[freq_col].unique()
    if len(conditions) == 0:
        conditions = sorted(df["condition"].unique())
    if len(conditions) < 6:
        colors = ["sienna", "limegreen", "royalblue", "darkviolet", "darkgray", "indianred"]
    else:
        colors = ["lightgreen", "limegreen", "darkgreen", "sandybrown", "chocolate", "saddlebrown",
                "cyan", "darkturquoise", "cadetblue", "orchid", "mediumorchid", "darkorchid"]

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
            if zoom:
                ax.set_ylim(0, 0.3)
            ax.legend()
            fig.tight_layout()
            if transparent:
                fig.patch.set_alpha(0.0)
            extra_filename = "_zoom" if zoom else ""
            fig.savefig(f"output/{exp_dir}/{dimension}pair_correlation_{pair}_{freq}{extra_filename}.png")


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
    df["total"] = df["sensitive"] + df["resistant"]
    df = bull_pc(df)

    # pair_correlation_functions(df, exp_dir, dimension, freq_col="time", freq_ticks=[0, 1000, 2000], transparent=True,
    #                             conditions=["sensitive_equal", "sensitive_agtb", "sensitive_bgta"])
    # pair_correlation_functions(df, exp_dir, dimension, freq_col="time", freq_ticks=[0, 1000, 2000], transparent=True,
    #                             conditions=["coexistence_equal", "coexistence_bgtc", "coexistence_cgtb"])
    pair_correlation_functions(df, exp_dir, dimension, freq_col="time", freq_ticks=[0, 1000, 2000], transparent=True)
    pair_correlation_functions(df, exp_dir, dimension, freq_col="time", freq_ticks=[1000, 2000], transparent=True, zoom=True)
    # pair_correlation_functions(df, exp_dir, dimension, freq_col="time", freq_ticks=[2000], transparent=True, zoom=True, 
    #                            conditions=["sensitive_equal", "sensitive_agtb", "sensitive_bgta", "resistant_equal", "resistant_cgtd", "resistant_dgtc"])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide an experiment directory and dimension.")