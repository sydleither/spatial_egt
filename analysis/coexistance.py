import os
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


def coexistance_plot(df, exp_dir):
    figure, axis = plt.subplots(2, 3, figsize=(10,12), dpi=150)
    row = 0
    col = 0
    generations = df["time"].unique()
    for condition in sorted(df["condition"].unique()):
        df_c = df.loc[df["condition"] == condition]
        plot_line(axis[row][col], df_c.loc[df_c["dimension"] == "2D"], generations, "null_resistant", "sienna", "2D")
        plot_line(axis[row][col], df_c.loc[df_c["dimension"] == "3D"], generations, "null_resistant", "limegreen", "3D")
        plot_line(axis[row][col], df_c.loc[df_c["dimension"] == "WM"], generations, "null_resistant", "turquoise", "WM")
        axis[row][col].set(xlabel="Time", ylabel="Resistant Cells", title=condition)
        axis[row][col].legend(framealpha=0.33)
        col += 1
        if col % 3 == 0:
            col = 0
            row += 1
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle("")
    plt.savefig(f"output/{exp_dir}/coexistence2.png", transparent=False)
    plt.close()


def read_all(exp_dir):
    df = pd.DataFrame(columns=["condition","dimension","rep","time","null_sensitive",
                               "null_resistant","continuous_sensitive","continuous_resistant"])
    exp_path = f"output/{exp_dir}"
    for exp_name in os.listdir(exp_path):
        run_path = f"{exp_path}/{exp_name}"
        if os.path.isfile(run_path):
                continue
        for rep_dir in os.listdir(run_path):
            rep_path = f"{run_path}/{rep_dir}"
            if os.path.isfile(rep_path):
                continue
            for result_file in os.listdir(rep_path):
                result_path = f"{rep_path}/{result_file}"
                if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
                    print(f"File not found: {result_path}")
                    continue
                df_i = pd.read_csv(result_path)
                df_i["rep"] = int(rep_dir)
                df_i["condition"] = exp_name
                df_i["dimension"] = result_file[0:2]
                df = pd.concat([df, df_i])
    return df


def main():
    df = read_all("coexist")
    coexistance_plot(df, "coexist")


if __name__ == "__main__":
    main()