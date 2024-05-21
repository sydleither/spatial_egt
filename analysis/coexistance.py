import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import read_all, plot_line


def coexistance_plot(df, exp_dir):
    figure, axis = plt.subplots(3, 4, figsize=(18,15), dpi=150)
    row = 0
    col = 0
    for condition in sorted(df["condition"].unique()):
        df_c = df.loc[df["condition"] == condition]
        plot_line(axis[row][col], df_c.loc[df_c["dimension"] == "2D"], "time", "fr", "sienna", "2D")
        plot_line(axis[row][col], df_c.loc[df_c["dimension"] == "3D"], "time", "fr", "green", "3D")
        plot_line(axis[row][col], df_c.loc[df_c["dimension"] == "WM"], "time", "fr", "steelblue", "WM")
        axis[row][col].set(xlabel="Time", ylabel="Proportion Resistant", title=condition)
        axis[row][col].legend(framealpha=0.33)
        col += 1
        if col % 4 == 0:
            col = 0
            row += 1
    figure.tight_layout()
    figure.suptitle("")
    plt.savefig(f"output/{exp_dir}/coexistence.png", transparent=False)
    plt.close()


def diff_between_dim(df, exp_dir):
    colors = ["sienna", "green", "steelblue"]
    figure, axis = plt.subplots(1, 1, figsize=(6,6), dpi=150)
    for i,dimension in enumerate(sorted(df["dimension"].unique())):
        df_dim = df.loc[df["dimension"] == dimension]
        cond_max_times = df_dim[["condition", "time"]].groupby("condition").max().reset_index()
        cond_max_times.set_index(["condition", "time"], inplace=True)
        df_dim.set_index(["condition", "time"], inplace=True)
        df_final = cond_max_times.merge(df_dim, on=["time", "condition"], how="left").reset_index()
        plot_line(axis, df_final, "condition", "fr", colors[i], dimension)
    axis.set(xlabel="Theoretical Percent Resistant", ylabel="Actual Proportion Resistant", title="End of Coexistence Experiments")
    axis.legend()
    figure.tight_layout()
    plt.savefig(f"output/{exp_dir}/coexist_line.png", transparent=False)
    plt.close()


def main():
    exp_name = "coexist"
    df = read_all(exp_name)
    df["null_total"] = df["null_resistant"] + df["null_sensitive"]
    df["condition"] = pd.to_numeric(df["condition"])
    df["fr"] = df["null_resistant"]/df["null_total"]

    coexistance_plot(df, exp_name)
    diff_between_dim(df, exp_name)


if __name__ == "__main__":
    main()