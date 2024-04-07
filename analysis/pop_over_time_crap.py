from csv import reader
import os
import sys

import matplotlib.pyplot as plt
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


def loadCSV(pathToFile, firstRowIsHeader=True):
    DATA = None
    HEADER = None
    with open(pathToFile,'r') as CSVFile:
        csvReader = reader(CSVFile)
        if firstRowIsHeader:
            firstRow = next(csvReader)
            HEADER = {columnName:columnIndex for columnIndex,columnName in enumerate(firstRow)}
        DATA = [[int(x) for x in row] for row in csvReader]
    return DATA, HEADER


def main(exp_dir, exp_name, dimension, transparent=False):
    col_to_idx = None
    replicates = []
    run_path = f"output/{exp_dir}/{exp_name}/"
    for rep_dir in os.listdir(run_path):
        if os.path.isfile(run_path+rep_dir):
            continue
        data_loc = f"output/{exp_dir}/{exp_name}/{rep_dir}/{dimension}populations.csv"
        if not os.path.exists(data_loc):
            print(f"File not found for rep {rep_dir}")
            continue
        rep_i, header_i = loadCSV(data_loc, True)
        if col_to_idx is None:
            col_to_idx = header_i
        replicates.append(transpose(rep_i[0:5]))

    experiment_names = [x for x in col_to_idx if x != "time"]
    max_pop = max([max([max(replicates[rep][col_to_idx[col]]) for col in experiment_names]) for rep in range(len(replicates))])

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2,3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    exp_types = list(set(x.split("_")[0] for x in experiment_names))
    axes = [ax1, ax2, ax3]
    plot_types = {exp_types[i]:axes[i] for i in range(len(exp_types))}
    for exp_i in range(len(experiment_names)):
        experiment_data = [replicates[rep][col_to_idx[experiment_names[exp_i]]] for rep in range(len(replicates))]
        exit()
        data_mean, lower, upper = calculate_confidence_interval(experiment_data)
        plot_types[exp_i].plot(replicates[0][col_to_idx["time"]], df["null_sensitive"], label="Sensitive", linewidth=2, color="sandybrown")

    #ax1.plot(df["time"], df["null_sensitive"], label="Sensitive", linewidth=2, color="sandybrown")
    #ax1.plot(df["time"], df["null_resistant"], label="Resistant", linewidth=2, color="saddlebrown")
    # condition = df.pivot(index='rep', columns='time', values='continuous_sensitive').values.tolist()
    # data_mean, lower, upper = calculate_confidence_interval(condition)
    # ax1.plot(data_mean, label="Sensitive", linewidth=2, color="sandybrown")
    # ax1.fill_between(range(len(data_mean)), lower, upper, alpha=0.5)
    # ax1.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="No Therapy")
    # ax1.legend()

    # ax2.plot(df["time"], df["adaptive_sensitive"], label="Sensitive", linewidth=2, color="lightpink")
    # ax2.plot(df["time"], df["adaptive_resistant"], label="Resistant", linewidth=2, color="deeppink")
    # ax2.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="Adaptive Therapy")
    # ax2.legend()

    # ax3.plot(df["time"], df["continuous_sensitive"], label="Sensitive", linewidth=2, color="lightgreen")
    # ax3.plot(df["time"], df["continuous_resistant"], label="Resistant", linewidth=2, color="darkgreen")
    # ax3.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="Continuous Therapy")
    # ax3.legend()

    # df["null_total"] = df["null_sensitive"] + df["null_resistant"]
    # df["adaptive_total"] = df["adaptive_sensitive"] + df["adaptive_resistant"]
    # df["continuous_total"] = df["continuous_sensitive"] + df["continuous_resistant"]
    # max_pop = df[["null_total", "adaptive_total", "continuous_total"]].max(axis=None)

    # ax4.plot(df["time"], df["null_total"], label="No Therapy (total)", linewidth=2, color="peru")
    # ax4.plot(df["time"], df["adaptive_total"], label="Adaptive (total)", linewidth=2, color="hotpink")
    # ax4.plot(df["time"], df["continuous_total"], label="Continuous (total)", linewidth=2, color="limegreen")
    # ax4.set(ylim=(0, max_pop), xlabel="Time", ylabel="Cells", title="Total Cells Over Time")
    # ax4.legend()

    # fig.suptitle(exp_name+" "+dimension)
    # fig.tight_layout()
    # if transparent:
    #     fig.patch.set_alpha(0.0)
    # fig.savefig(f"output/{exp_dir}/{exp_name}/{dimension}pop_over_time.png")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide an experiment name and dimension.")