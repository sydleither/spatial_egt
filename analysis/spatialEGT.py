from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams['axes.prop_cycle'] = cycler(color=["#ef7c8e", "#4c956c", "#d68c45"])


def main():
    df = pd.read_csv("SpatialEGTPopulations.csv")

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.plot(df["time"], df["adaptive_sensitive"], label="Sensitive")
    ax1.plot(df["time"], df["adaptive_resistant"], label="Resistant")
    ax1.set(ylim=(0, 3000), xlabel="Time", ylabel="Cells", title="Adaptive Therapy")
    ax1.legend()

    ax2.plot(df["time"], df["continuous_sensitive"], label="Sensitive")
    ax2.plot(df["time"], df["continuous_resistant"], label="Resistant")
    ax2.set(ylim=(0, 3000), xlabel="Time", ylabel="Cells", title="Continuous Therapy")
    ax2.legend()

    ax3.plot(df["time"], df["adaptive_sensitive"]+df["adaptive_resistant"], label="Adaptive")
    ax3.plot(df["time"], df["continuous_sensitive"]+df["continuous_resistant"], label="Continuous")
    ax3.set(ylim=(0, 3000), xlabel="Time", ylabel="Cells", title="Total Cells Over Time")
    ax3.legend()

    fig.tight_layout()
    fig.savefig("SpatialEGTPopOverTime.png")


main()