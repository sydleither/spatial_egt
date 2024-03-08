import sys

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams['axes.prop_cycle'] = cycler(color=["#ef7c8e", "#4c956c", "#d68c45"])


def main(exp_name):
    df = pd.read_csv(f"output/{exp_name}/populations.csv")
    max_pop = df.drop("time", axis=1).max(axis=None)

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.plot(df["time"], df["adaptive_sensitive"], label="Sensitive")
    ax1.plot(df["time"], df["adaptive_resistant"], label="Resistant")
    ax1.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="Adaptive Therapy")
    ax1.legend()

    ax2.plot(df["time"], df["continuous_sensitive"], label="Sensitive")
    ax2.plot(df["time"], df["continuous_resistant"], label="Resistant")
    ax2.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="Continuous Therapy")
    ax2.legend()

    df["adaptive_total"] = df["adaptive_sensitive"] + df["adaptive_resistant"]
    df["continuous_total"] = df["continuous_sensitive"] + df["continuous_resistant"]
    max_pop = df[["adaptive_total", "continuous_total"]].max(axis=None)

    ax3.plot(df["time"], df["adaptive_total"], label="Adaptive")
    ax3.plot(df["time"], df["continuous_total"], label="Continuous")
    ax3.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="Total Cells Over Time")
    ax3.legend()

    fig.suptitle(exp_name)
    fig.tight_layout()
    fig.savefig(f"output/{exp_name}/pop_over_time.png")



if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide an experiment name.")