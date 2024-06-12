import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

from common import read_dim, plot_line


def main(exp_dir, exp_name, dimension, transparent=False):
    df = read_dim(exp_dir, exp_name, dimension)
    df["total"] = df["sensitive"] + df["resistant"]
    models = df["model"].unique()
    num_models = len(models)

    s_colors = ["sandybrown", "lightpink", "lightgreen"]
    r_colors = ["saddlebrown", "deeppink", "darkgreen"]
    a_colors = ["sienna", "hotpink", "limegreen"]

    ymin = 0
    ymax = df["total"].max()

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, num_models)
    ax_all = fig.add_subplot(gs[1, :])

    for i in range(num_models):
        ax = fig.add_subplot(gs[0, i])
        df_model = df.loc[df["model"] == models[i]]
        plot_line(ax, df_model, "time", "sensitive", s_colors[i], "Sensitive")
        plot_line(ax, df_model, "time", "resistant", r_colors[i], "Resistant")
        plot_line(ax_all, df_model, "time", "total", a_colors[i], models[i])
        ax.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Cells", title=models[i])
        ax.legend()
    ax_all.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Cells", title="Total Cells Over Time")
    ax_all.legend()

    fig.suptitle(exp_name)
    fig.tight_layout()
    if transparent:
        fig.patch.set_alpha(0.0)
    fig.savefig(f"output/{exp_dir}/{exp_name}/{dimension}pop_over_time.png")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide an experiment directory, name, and dimension.")