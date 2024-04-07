import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import sem, t


def transpose(l):
    return list(zip(*l))


def calculate_confidence_interval(data):
    num_across = len(data[0])
    data_t = list(zip(*data))
    data_mean = list(map(np.mean, data_t))
    intervals = [t.interval(confidence=0.95, df=len(data_t[i])-1, loc=data_mean[i], scale=sem(data_t[i])) for i in range(num_across)]
    lower = [intervals[i][0] for i in range(num_across)]
    upper = [intervals[i][1] for i in range(num_across)]
    return data_mean, lower, upper


def plot_line(ax, df, generations, condition_name, color, label):
    condition = df.pivot(index='rep', columns='time', values=condition_name).values.tolist()
    data_mean, lower, upper = calculate_confidence_interval(condition)
    ax.plot(generations, data_mean, label=label, linewidth=2, color=color)
    ax.fill_between(list(generations), lower, upper, color=color, alpha=0.5)


def main(exp_dir, exp_name, dimension, transparent=False):
    df = pd.DataFrame(columns=["rep","time","null_sensitive","null_resistant","adaptive_sensitive",
                               "adaptive_resistant","continuous_sensitive","continuous_resistant"])
    run_path = f"output/{exp_dir}/{exp_name}/"
    for rep_dir in os.listdir(run_path):
        if os.path.isfile(run_path+rep_dir):
            continue
        data_loc = f"output/{exp_dir}/{exp_name}/{rep_dir}/{dimension}populations.csv"
        if not os.path.exists(data_loc):
            print(f"File not found for rep {rep_dir}")
            continue
        df_i = pd.read_csv(data_loc)
        df_i["rep"] = int(rep_dir)
        df = pd.concat([df, df_i])

    max_pop = df.drop(["time","rep"], axis=1).max(axis=None)
    generations = df["time"].unique()

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2,3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    plot_line(ax1, df, generations, "null_sensitive", "sandybrown", "Sensitive")
    plot_line(ax1, df, generations, "null_resistant", "saddlebrown", "Resistant")
    ax1.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="No Therapy")
    ax1.legend()

    plot_line(ax2, df, generations, "adaptive_sensitive", "lightpink", "Sensitive")
    plot_line(ax2, df, generations, "adaptive_resistant", "deeppink", "Resistant")
    ax2.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="Adaptive Therapy")
    ax2.legend()

    plot_line(ax3, df, generations, "continuous_sensitive", "lightgreen", "Sensitive")
    plot_line(ax3, df, generations, "continuous_resistant", "darkgreen", "Resistant")
    ax3.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="Continuous Therapy")
    ax3.legend()

    df["null_total"] = df["null_sensitive"] + df["null_resistant"]
    df["adaptive_total"] = df["adaptive_sensitive"] + df["adaptive_resistant"]
    df["continuous_total"] = df["continuous_sensitive"] + df["continuous_resistant"]
    min_pop = df[["null_total", "adaptive_total", "continuous_total"]].min(axis=None)
    max_pop = df[["null_total", "adaptive_total", "continuous_total"]].max(axis=None)

    plot_line(ax4, df, generations, "null_total", "sienna", "Null")
    plot_line(ax4, df, generations, "adaptive_total", "hotpink", "Adaptive")
    plot_line(ax4, df, generations, "continuous_total", "limegreen", "Continuous")
    ax4.set(ylim=(min_pop, max_pop), xlabel="Time", ylabel="Cells", title="Total Cells Over Time")
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
        print("Please provide an experiment name and dimension.")