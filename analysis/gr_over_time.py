import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

from common import plot_line, read_specific


def main(exp_dir, exp_name, dimension, transparent):
    df = read_specific(exp_dir, exp_name, dimension, "populations")
    df = df.reset_index()
    df_gr = df.drop(["index", "time", "rep", "model"], axis=1)
    df_gr = df_gr.rolling(window=2).apply(lambda x: x.values[1]/x.values[0] if x.values[0] != 0 else 0)
    df = df_gr.join(df[["time", "rep", "model"]])
    df["diff"] = df["sensitive"] - df["resistant"]
    models = df["model"].unique()
    num_models = len(models)

    s_colors = ["sandybrown", "lightpink", "lightgreen"]
    r_colors = ["saddlebrown", "deeppink", "darkgreen"]
    a_colors = ["sienna", "hotpink", "limegreen"]

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, num_models)
    ax_all = fig.add_subplot(gs[1, :])

    for i in range(num_models):
        ax = fig.add_subplot(gs[0, i])
        df_model = df.loc[df["model"] == models[i]]
        plot_line(ax, df_model, "time", "sensitive", s_colors[i], "Sensitive")
        plot_line(ax, df_model, "time", "resistant", r_colors[i], "Resistant")
        plot_line(ax_all, df_model, "time", "diff", a_colors[i], models[i])
        ax.axhline(1, linestyle="dashed", color="black", linewidth=1)
        ax.set(xlabel="Time", ylabel="Cells", title=models[i])
        ax.legend()
    ax_all.axhline(0, linestyle="dashed", color="black", linewidth=1)
    ax_all.set(xlabel="Time", ylabel="Cells", title="Difference in Growth Rate")
    ax_all.legend()

    fig.suptitle(f"{exp_name} Growth Rates")
    fig.tight_layout()
    if transparent == "true":
        fig.patch.set_alpha(0.0)
    fig.savefig(f"output/{exp_dir}/{exp_name}/{dimension}gr_over_time.png")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Please provide an experiment directory, name, dimension, and transparency (true).")