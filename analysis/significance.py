from itertools import combinations
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SIM_TYPES = ["null", "continuous"]


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
                if not os.path.exists(result_path):
                    print(f"File not found: {result_path}")
                    continue
                df_i = pd.read_csv(result_path)
                df_i["rep"] = int(rep_dir)
                df_i["condition"] = exp_name
                df_i["dimension"] = result_file[0:2]
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


def significance_table(df, group_type, groups, sim_type):
    df_et = pd.DataFrame()
    for group in groups:
        df_group = df.loc[df[group_type] == group]
        df_et[group] = get_extinction_times(df_group, sim_type)

    df_p = pd.DataFrame(index=groups, columns=groups)
    pairs = list(combinations(groups, 2))
    for pair in pairs:
        p = permutation_test(df_et[pair[0]], df_et[pair[1]])
        df_p.at[pair[0], pair[1]] = p
        df_p.at[pair[1], pair[0]] = p
    print(df_p)
    print()


def extinction_plot(df, exp_dir, group_type, groups, fixed_var, sim_types):
    df_ets = []
    for sim_type in sim_types:
        for group in groups:
            df_et_i = pd.DataFrame()
            df_cond = df.loc[df[group_type] == group]
            df_et_i["extinction_time"] = get_extinction_times(df_cond, sim_type)
            df_et_i[group_type] = group
            df_et_i["sim_type"] = "No Drug" if sim_type == "null" else "Drug"
            df_ets.append(df_et_i)
    df_et_all = pd.concat(df_ets)

    figure, axis = plt.subplots(1, 1, figsize=(8,5), dpi=150)
    x = sns.boxplot(data=df_et_all, x="sim_type", y="extinction_time", hue=group_type, 
                    ax=axis, palette="Set2")
    x.legend(framealpha=0.33, title=group_type)
    x.set(xlabel="Type of Simulation")
    x.set(ylabel="Time of Extinction")
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle(f"Time of Extinction of Either Cell Line in {fixed_var} Experiments")
    plt.savefig(f"output/{exp_dir}/{fixed_var}_extinction_times.png", transparent=True)
    plt.close()


def extinction_plot_generic(df, exp_dir, group1_type, group2_type, sim_type):
    df_ets = []
    for group1 in sorted(df[group1_type].unique()):
        for group2 in sorted(df[group2_type].unique()):
            df_et_i = pd.DataFrame()
            df_cond = df.loc[(df[group1_type] == group1) & (df[group2_type] == group2)]
            df_et_i["extinction_time"] = get_extinction_times(df_cond, sim_type)
            df_et_i[group1_type] = group1
            df_et_i[group2_type] = group2
            df_ets.append(df_et_i)
    df_et_all = pd.concat(df_ets)

    sim_type_clean = "No Drug" if sim_type == "null" else "Drug"

    figure, axis = plt.subplots(1, 1, figsize=(8,5), dpi=150)
    x = sns.boxplot(data=df_et_all, x=group1_type, y="extinction_time", hue=group2_type, 
                    ax=axis, palette="Set2")
    x.legend(framealpha=0.33, title=group2_type)
    x.set(xlabel=group1_type)
    x.set(ylabel="Time of Extinction")
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle(f"Time of Extinction of Either Cell Line in {sim_type_clean} Experiments")
    plt.savefig(f"output/{exp_dir}/{group1_type}_{group2_type}_{sim_type}_extinction_times.png", transparent=True)
    plt.close()


def main_across_sim(exp_dir, analysis_type, fixed_type, fixed_var):
    df = read_all(exp_dir)
    df_fixed = df.loc[df[fixed_type] == fixed_var]

    groups = sorted(df_fixed[analysis_type].unique())
    for sim_type in SIM_TYPES:
        print(sim_type)
        significance_table(df_fixed, analysis_type, groups, sim_type)
    extinction_plot(df_fixed, exp_dir, analysis_type, groups, fixed_var, SIM_TYPES)


def main_fixed_sim(exp_dir, group1, group2, sim_type):
    df = read_all(exp_dir)
    for group1x in sorted(df[group1].unique()):
        df_fixed = df.loc[df[group1] == group1x]
        print(group1x)
        significance_table(df_fixed, group2, sorted(df[group2].unique()), sim_type)
    extinction_plot_generic(df, exp_dir, group1, group2, sim_type)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        if sys.argv[4] in SIM_TYPES:
            main_fixed_sim(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            main_across_sim(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Please provide an experiment directory, type of analysis, "+ 
              "the fixed variable type, and the fixed variable.")
        print("Or provide an experiment directory, group1, group2, and sim type.")
        print("Examples:")
        print("\tpython3 analysis/significance.py games dimension condition coexistance")
        print("\tpython3 analysis/significance.py games condition dimension 2D")
        print("\tpython3 analysis/significance.py games condition dimension null")