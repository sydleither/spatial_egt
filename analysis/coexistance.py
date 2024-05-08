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
        plot_line(axis, df_dim.loc[df_dim["time"] == df_dim["time"].max()], "condition", "fr", colors[i], dimension)
    axis.set(xlabel="Theoretical Percent Resistant", ylabel="Actual Proportion Resistant", title="End of Coexistence Experiments")
    axis.legend()
    figure.tight_layout()
    plt.savefig(f"output/{exp_dir}/coexist_line.png", transparent=False)
    plt.close()


def diff_between_dim_noerror(df, exp_dir):
    fr_2d = {}
    fr_3d = {}
    fr_wm = {}
    for condition in sorted(df["condition"].unique()):
        print(condition)
        df_c = df.loc[df["condition"] == condition]
        for dimension in sorted(df_c["dimension"].unique()):
            df_cd = df_c.loc[df_c["dimension"] == dimension]
            avg_end_fr = np.mean(df_cd.loc[df_cd["time"] == df_cd["time"].max()]["null_resistant"].to_list())
            avg_end_total = np.mean(df_cd.loc[df_cd["time"] == df_cd["time"].max()]["null_total"].to_list())
            fr = avg_end_fr/avg_end_total
            if dimension == "2D":
                fr_2d[int(condition)] = fr
            elif dimension == "3D":
                fr_3d[int(condition)] = fr
            elif dimension == "WM":
                fr_wm[int(condition)] = fr
            print(f"\t{dimension}   {round(fr, 3)}")

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(fr_2d.keys(), fr_2d.values(), label="2D", color="sienna")
    ax.plot(fr_3d.keys(), fr_3d.values(), label="3D", color="green")
    ax.plot(fr_wm.keys(), fr_wm.values(), label="WM", color="steelblue")
    ax.set(xlim=(10, 95), xlabel="Theoretical Percent Resistant", ylabel="Actual Proportion Resistant", title="End of Coexistence Experiments")
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"output/{exp_dir}/coexist_line2.png", transparent=False)
    plt.close()


def main():
    df = read_all("coexist")
    df["null_total"] = df["null_resistant"] + df["null_sensitive"]
    df["condition"] = pd.to_numeric(df["condition"])
    df["fr"] = df["null_resistant"]/df["null_total"]

    coexistance_plot(df, "coexist")
    diff_between_dim(df, "coexist")


if __name__ == "__main__":
    main()