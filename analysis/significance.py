from itertools import combinations
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_all(exp_dir, dimension):
    df = pd.DataFrame(columns=["condition","rep","time","null_sensitive","null_resistant",
                               "continuous_sensitive","continuous_resistant"])
    exp_path = f"output/{exp_dir}/"
    for exp_name in os.listdir(exp_path):
        run_path = f"output/{exp_dir}/{exp_name}"
        if os.path.isfile(run_path):
                continue
        for rep_dir in os.listdir(run_path):
            if os.path.isfile(run_path+"/"+rep_dir):
                continue
            data_loc = f"{run_path}/{rep_dir}/{dimension}populations.csv"
            if not os.path.exists(data_loc):
                print(f"File not found for rep {rep_dir}")
                continue
            df_i = pd.read_csv(data_loc)
            df_i["rep"] = int(rep_dir)
            df_i["condition"] = exp_name
            df = pd.concat([df, df_i])
    return df


def get_extinction_times(df, sim_type):
    extinction_times = []
    for rep in df["rep"].unique():
        df_rep = df.loc[df["rep"] == rep]
        extinct = df_rep.loc[(df_rep[f"{sim_type}_resistant"] == 0) | (df_rep[f"{sim_type}_sensitive"] == 0)]["time"].tolist()
        extinction_times.append(extinct[0] if len(extinct) > 0 else 0)
    return extinction_times


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


def significance_table(df, conditions, sim_types):
    for sim_type in sim_types:
        df_et = pd.DataFrame()
        for condition in conditions:
            df_cond = df.loc[df["condition"] == condition]
            df_et[condition] = get_extinction_times(df_cond, sim_type)

        df_p = pd.DataFrame(index=conditions, columns=conditions)
        pairs = list(combinations(conditions, 2))
        for pair in pairs:
            p = permutation_test(df_et[pair[0]], df_et[pair[1]])
            df_p.at[pair[0], pair[1]] = p
            df_p.at[pair[1], pair[0]] = p
        print(sim_type)
        print(df_p)
        print()


def extinction_plot(df, exp_dir, dimension, conditions, sim_types):
    df_ets = []
    for sim_type in sim_types:
        for condition in conditions:
            df_et_i = pd.DataFrame()
            df_cond = df.loc[df["condition"] == condition]
            df_et_i["extinction_time"] = get_extinction_times(df_cond, sim_type)
            df_et_i["condition"] = condition
            df_et_i["sim_type"] = "No Drug" if sim_type == "null" else "Drug"
            df_ets.append(df_et_i)
    df_et_all = pd.concat(df_ets)

    figure, axis = plt.subplots(1, 1, figsize=(8,5), dpi=150)
    x = sns.boxplot(data=df_et_all, x="sim_type", y="extinction_time", hue="condition", 
                    ax=axis, palette=sns.color_palette(["#f4a9b5", "#d68c45", "#4c956c"]))
    x.legend(framealpha=0.33, title="Game Space")
    x.set(xlabel="Type of Simulation")
    x.set(ylabel="Time of Extinction")
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle(f"Time of Extinction of Either Cell Line in {dimension} Experiments")
    plt.savefig(f"output/{exp_dir}/{dimension}_extinction_times.png", transparent=False)
    plt.close()


def main(exp_dir, dimension):
    df = read_all(exp_dir, dimension)
    conditions = sorted(df["condition"].unique())
    sim_types = ["null", "continuous"]

    significance_table(df, conditions, sim_types)
    extinction_plot(df, exp_dir, dimension, conditions, sim_types)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide an experiment directory and dimension.")