import sys

import matplotlib.pyplot as plt
import pandas as pd


def main(exp_dir, exp_name, dimension, transparent=False):
    df = pd.read_csv(f"output/{exp_dir}/{exp_name}/{dimension}populations.csv")
    max_pop = df.drop("time", axis=1).max(axis=None)

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2,3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    ax1.plot(df["time"], df["null_sensitive"], label="Sensitive", linewidth=2, color="sandybrown")
    ax1.plot(df["time"], df["null_resistant"], label="Resistant", linewidth=2, color="saddlebrown")
    ax1.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="No Therapy")
    ax1.legend()

    ax2.plot(df["time"], df["adaptive_sensitive"], label="Sensitive", linewidth=2, color="lightpink")
    ax2.plot(df["time"], df["adaptive_resistant"], label="Resistant", linewidth=2, color="deeppink")
    ax2.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="Adaptive Therapy")
    ax2.legend()

    ax3.plot(df["time"], df["continuous_sensitive"], label="Sensitive", linewidth=2, color="lightgreen")
    ax3.plot(df["time"], df["continuous_resistant"], label="Resistant", linewidth=2, color="darkgreen")
    ax3.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="Continuous Therapy")
    ax3.legend()

    df["null_total"] = df["null_sensitive"] + df["null_resistant"]
    df["adaptive_total"] = df["adaptive_sensitive"] + df["adaptive_resistant"]
    df["continuous_total"] = df["continuous_sensitive"] + df["continuous_resistant"]
    max_pop = df[["null_total", "adaptive_total", "continuous_total"]].max(axis=None)

    ax4.plot(df["time"], df["null_total"], label="Null", linewidth=2, color="sienna")
    ax4.plot(df["time"], df["adaptive_total"], label="Adaptive", linewidth=2, color="hotpink")
    ax4.plot(df["time"], df["continuous_total"], label="Continuous", linewidth=2, color="limegreen")
    ax4.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="Total Cells Over Time")
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