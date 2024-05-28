import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

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
    
    df = df.reset_index()
    df_gr = df.drop(["time", "rep", "index"], axis=1).rolling(window=2).apply(lambda x: x.values[1]/x.values[0] if x.values[0] != 0 else 0)
    df_gr["null_diff"] = df_gr["null_sensitive"] - df_gr["null_resistant"]
    df_gr["adaptive_diff"] = df_gr["adaptive_sensitive"] - df_gr["adaptive_resistant"]
    df_gr["continuous_diff"] = df_gr["continuous_sensitive"] - df_gr["continuous_resistant"]
    df_gr = df_gr.join(df[["time", "rep"]])

    ymin = 0
    ymax = 1.1

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2,3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    plot_line(ax1, df_gr, "time", "null_sensitive", "sandybrown", "Sensitive")
    plot_line(ax1, df_gr, "time", "null_resistant", "saddlebrown", "Resistant")
    ax1.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Growth Rate", title="No Therapy")
    ax1.legend()

    plot_line(ax2, df_gr, "time", "adaptive_sensitive", "lightpink", "Sensitive")
    plot_line(ax2, df_gr, "time", "adaptive_resistant", "deeppink", "Resistant")
    ax2.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Growth Rate", title="Adaptive Therapy")
    ax2.legend()

    plot_line(ax3, df_gr, "time", "continuous_sensitive", "lightgreen", "Sensitive")
    plot_line(ax3, df_gr, "time", "continuous_resistant", "darkgreen", "Resistant")
    ax3.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Growth Rate", title="Continuous Therapy")
    ax3.legend()

    plot_line(ax4, df_gr, "time", "null_diff", "sienna", "Null")
    plot_line(ax4, df_gr, "time", "adaptive_diff", "hotpink", "Adaptive")
    plot_line(ax4, df_gr, "time", "continuous_diff", "limegreen", "Continuous")
    ax4.set(ylim=(-1.1, 1.1), xlabel="Time", ylabel="Growth Rate", title="Difference in Growth Rate")
    ax4.legend()

    fig.suptitle(exp_name)
    fig.tight_layout()
    if transparent:
        fig.patch.set_alpha(0.0)
    fig.savefig(f"output/{exp_dir}/{exp_name}/{dimension}gr_over_time.png")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide an experiment directory, name, and dimension.")