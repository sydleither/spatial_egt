import os

import numpy as np
import pandas as pd
from scipy.stats import sem, t


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
        for rep_dir in os.listdir(run_path):
            rep_path = f"{run_path}/{rep_dir}"
            if os.path.isfile(rep_path):
                continue
            for result_file in os.listdir(rep_path):
                result_path = f"{rep_path}/{result_file}"
                if not result_file.endswith(f"{file_name}.csv") or not result_file.startswith(dimension):
                    continue
                if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
                    print(f"File not found: {result_path}")
                    continue
                df_i = pd.read_csv(result_path)
                df_i["rep"] = int(rep_dir)
                df_i["condition"] = exp_name
                df_i["dimension"] = result_file[0:2]
                df = pd.concat([df, df_i])
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