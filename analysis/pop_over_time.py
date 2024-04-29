import os
import sys
import warnings
warnings.filterwarnings("ignore")

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


def main(exp_dir, exp_name, dimension, transparent=True):
    df = pd.DataFrame(columns=["rep","time","null_sensitive","null_resistant",
                               "continuous_sensitive","continuous_resistant"])
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
    generations = df["time"].unique()
    df["null_total"] = df["null_sensitive"] + df["null_resistant"]
    df["continuous_total"] = df["continuous_sensitive"] + df["continuous_resistant"]

    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    plot_line(ax1, df, generations, "null_sensitive", "sandybrown", "Sensitive")
    plot_line(ax1, df, generations, "null_resistant", "saddlebrown", "Resistant")
    ax1.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Cells", title="Absence of Drug")
    ax1.legend(framealpha=0.33)
    if exp_name == "coexistance" and dimension == "WM":
        ax1.axhline(y=2232, linestyle="dotted", color="black")

    plot_line(ax2, df, generations, "continuous_sensitive", "lightgreen", "Sensitive")
    plot_line(ax2, df, generations, "continuous_resistant", "darkgreen", "Resistant")
    ax2.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Cells", title="Presence of Drug")
    ax2.legend(framealpha=0.33)

    plot_line(ax3, df, generations, "null_total", "sienna", "Absence of Drug")
    plot_line(ax3, df, generations, "continuous_total", "limegreen", "Presence of Drug")
    ax3.set(ylim=(ymin, ymax), xlabel="Time", ylabel="Cells", title="Total Cells Over Time")
    ax3.legend(framealpha=0.33)

    fig.suptitle(exp_name)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/{exp_name}/{dimension}pop_over_time.png", transparent=transparent)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide an experiment directory, name, and dimension.")