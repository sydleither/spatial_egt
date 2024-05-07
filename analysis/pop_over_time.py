import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from common import plot_line


def main(exp_dir, exp_name, dimension, transparent=False):
    df = pd.DataFrame()
    run_path = f"output/{exp_dir}/{exp_name}/"
    for rep_dir in os.listdir(run_path):
        if os.path.isfile(run_path+rep_dir):
            continue
        data_loc = f"output/{exp_dir}/{exp_name}/{rep_dir}/{dimension}populations.csv"
        if not os.path.exists(data_loc) or os.path.getsize(data_loc) == 0:
            print(f"File not found for rep {rep_dir}")
            continue
        df_i = pd.read_csv(data_loc)
        df_i["rep"] = int(rep_dir)
        df = pd.concat([df, df_i])

    ymin = 0
    ymax = 16000
    df["null_total"] = df["null_sensitive"] + df["null_resistant"]
    df["adaptive_total"] = df["adaptive_sensitive"] + df["adaptive_resistant"]
    df["continuous_total"] = df["continuous_sensitive"] + df["continuous_resistant"]

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2,3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    plot_line(ax1, df, "time", "null_sensitive", "sandybrown", "Sensitive")
    plot_line(ax1, df, "time", "null_resistant", "saddlebrown", "Resistant")
    ax1.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Cells", title="No Therapy")
    ax1.legend()

    plot_line(ax2, df, "time", "adaptive_sensitive", "lightpink", "Sensitive")
    plot_line(ax2, df, "time", "adaptive_resistant", "deeppink", "Resistant")
    ax2.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Cells", title="Adaptive Therapy")
    ax2.legend()

    plot_line(ax3, df, "time", "continuous_sensitive", "lightgreen", "Sensitive")
    plot_line(ax3, df, "time", "continuous_resistant", "darkgreen", "Resistant")
    ax3.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Cells", title="Continuous Therapy")
    ax3.legend()

    plot_line(ax4, df, "time", "null_total", "sienna", "Null")
    plot_line(ax4, df, "time", "adaptive_total", "hotpink", "Adaptive")
    plot_line(ax4, df, "time", "continuous_total", "limegreen", "Continuous")
    ax4.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Cells", title="Total Cells Over Time")
    ax4.legend()

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