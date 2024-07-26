import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import read_all, read_specific, plot_line


def get_times_to_progression(df):
    #calculate at which cell count the tumor is considered progressed
    df_p = df.groupby(["model", "condition", "dimension"]).first().reset_index()
    df_p["progression_amt"] = 1.2*df_p["total"]
    df_p = df_p[["model", "condition", "dimension", "progression_amt"]]
    df = df.merge(df_p, on=["model", "condition", "dimension"])

    #calculate at which time the tumor progressed (NaN for non-progressed tumors)
    df_pt = df.loc[df["total"] >= df["progression_amt"]]
    df_pt = df_pt.groupby(["model", "condition", "rep", "dimension"]).first().reset_index()
    df2 = df.groupby(["model", "condition", "rep", "dimension"]).first().reset_index()
    df2 = df2[["model", "condition", "rep", "dimension"]]
    df_pt = df2.merge(df_pt, on=["model", "condition", "rep", "dimension"], how="outer")
    return df_pt


def create_heatmaps_avg(exp_dir, df, categories=["threshold", "fR", "initial_density"], models=["adaptive", "continuous", "nodrug"]):
    for i in range(len(categories)):
        for j in range(i+1, len(categories)):
            col = categories[i]
            row = categories[j]
            fig, ax = plt.subplots(1, len(models), figsize=(len(models)*8, 8))
            for m in range(len(models)):
                df_avg = df.loc[df["model"] == models[m]].groupby([col, row])["time"].mean().unstack()
                sns.heatmap(df_avg, fmt="g", annot=True, cmap="Greens", ax=ax[m])
                ax[m].set(title=models[m])
            fig.patch.set_alpha(0.0)
            plt.savefig(f"output/{exp_dir}/heatmap_{col}_{row}.png")
            plt.close()


def create_heatmaps(exp_dir, df, categories=["threshold", "fR", "initial_density"], model="adaptive"):
        if len(categories) != 3:
            print("Unhardcode the heatmap function")
            return
        df = df.loc[df["model"] == model]
        col = categories[0]
        row = categories[1]
        across = categories[2]
        fig, ax = plt.subplots(1, len(categories), figsize=(len(categories)*8, 8))
        for i,x in enumerate(df[across].unique()):
            df_k = df.loc[df[across] == x]
            df_avg = df_k.groupby([col, row])["time"].mean().unstack()
            sns.heatmap(df_avg, fmt="g", annot=True, cmap="Greens", ax=ax[i])
            ax[i].set(title=f"{across} {x}")
        fig.patch.set_alpha(0.0)
        plt.savefig(f"output/{exp_dir}/{model}_heatmap_{col}_{row}_{across}.png")
        plt.close()


def main(exp_dir, dimension):
    thresholds = {"0":0.3, "1":0.5, "2":0.7}
    frs = {"0":0.01, "1":0.05, "2":0.1}
    cells = {"0":1875, "1":6250, "2":11250}
    df = read_all(exp_dir, "populations", dimension)
    df["total"] = df["sensitive"] + df["resistant"]
    df["threshold"] = df["condition"].str.split("_").str[1].str[-1].map(thresholds)
    df["fR"] = df["condition"].str.split("_").str[2].str[-1].map(frs)
    df["initial_density"] = df["condition"].str.split("_").str[3].str[-1].map(cells)
    df = get_times_to_progression(df)
    create_heatmaps_avg(exp_dir, df)
    create_heatmaps(exp_dir, df, categories=["threshold", "fR", "initial_density"])
    create_heatmaps(exp_dir, df, categories=["fR", "initial_density", "threshold"])
    create_heatmaps(exp_dir, df, categories=["initial_density", "threshold", "fR"])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide an experiment directory and dimension.")