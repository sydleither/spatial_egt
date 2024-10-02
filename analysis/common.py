import os

import numpy as np
import pandas as pd
from scipy.stats import sem, t
pd.options.mode.chained_assignment = None


def read_specific(exp_dir, exp_name, dimension, file_name):
    df = pd.DataFrame()
    run_path = f"output/{exp_dir}/{exp_name}"
    for rep_dir in os.listdir(run_path):
        rep_path = f"{run_path}/{rep_dir}"
        if os.path.isfile(rep_path):
            continue
        result_file = f"{rep_path}/{dimension}{file_name}.csv"
        if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
            print(f"File not found for rep {rep_dir}")
            continue
        df_i = pd.read_csv(result_file)
        df_i["dimension"] = dimension
        df_i["rep"] = int(rep_dir)
        df = pd.concat([df, df_i])
    return df


def read_all(exp_dir, file_name, dimension=""):
    df = pd.DataFrame()
    exp_path = f"output/{exp_dir}"
    for exp_name in os.listdir(exp_path):
        run_path = f"{exp_path}/{exp_name}"
        if os.path.isfile(run_path):
                continue
        df_i = read_specific(exp_dir, exp_name, dimension, file_name)
        df_i["condition"] = exp_name
        df = pd.concat([df, df_i])
    return df


def process_fs(df, key):
    df = df.loc[(df["fs"] > 0) & (df["radius"] <= 5)]
    df["weighted_fs"] = df["fs"]*df["total"]
    df_grp = df[key+["total", "weighted_fs"]].groupby(key).sum().reset_index()
    df_grp = df_grp.rename(columns={"total":"total_boundary", "weighted_fs":"weighted_fs_sum"})
    df = df.merge(df_grp, on=key)
    df["average_fs"] = df["weighted_fs_sum"] / df["total_boundary"]
    df["normalized_total"] = df["total"] / df["total_boundary"]
    return df


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


def plot_line(ax, df, x_col, y_col, color, label=None, avg_over="rep"):
    x_data = sorted(df[x_col].unique())
    condition = df.pivot(index=avg_over, columns=x_col, values=y_col).values.tolist()
    data_mean, lower, upper = calculate_confidence_interval(condition)
    ax.plot(x_data, data_mean, label=label, linewidth=2, color=color)
    ax.fill_between(list(x_data), lower, upper, color=color, alpha=0.5)


def permutation_test(data1, data2, num_permutations=1000, verbose=False):
    observed_mean = abs(np.mean(data1) - np.mean(data2))
    num_reps = len(data1)
    if num_reps != len(data2):
        print("Error in permutation test: different number of replicates between groups.")
        return
    
    combined = np.concatenate((data1, data2))
    permutated_means = []
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        resplit_groups = [combined[:num_reps], combined[num_reps:]]
        permutated_mean = abs(np.mean(resplit_groups[0]) - np.mean(resplit_groups[1]))
        permutated_means.append(permutated_mean)

    p = (permutated_means >= observed_mean).sum() / num_permutations
    if verbose:
        print(f"{data1.name}  {data2.name}")
        print(f"\tp-value: {p}")
        print(f"\tObserved mean diff: {observed_mean}")
        print(f"\tAverage permutated mean diff: {np.mean(permutated_means)}")
    return p


def get_colors():
    return ["#509154", "#A9561E", "#77BCFD", "#B791D4", "#EEDD5D",
            "#738696", "#24BCA8", "#D34A4F", "#8D81FE", "#FDA949"]