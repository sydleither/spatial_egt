import sys

from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import read_all


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


def create_heatmaps_rel(exp_dir, df, categories=["threshold", "fR", "initial_density"]):
        if len(categories) != 3:
            print("Unhardcode the heatmap function")
            return
        col = categories[0]
        row = categories[1]
        across = categories[2]

        fig, ax = plt.subplots(1, len(categories), figsize=(len(categories)*8, 8))
        cmap = plt.get_cmap("PiYG")
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)
        bounds = [-50, -40, -30, -20, -10, -1, 1, 10, 20, 30, 40, 50]
        norm = BoundaryNorm(bounds, cmap.N)

        for i,x in enumerate(df[across].unique()):
            df_k = df.loc[df[across] == x]
            df_avg_a = df_k.loc[df_k["model"] == "adaptive"].groupby([col, row])["time"].mean().unstack()
            df_avg_c = df_k.loc[df_k["model"] == "continuous"].groupby([col, row])["time"].mean().unstack()
            df_avg = df_avg_a.divide(df_avg_c)
            df_avg = 100*(df_avg-1)
            sns.heatmap(df_avg, fmt="g", annot=True, cmap=cmap, norm=norm, ax=ax[i])

        fig.patch.set_alpha(0.0)
        fig.tight_layout()
        plt.savefig(f"output/{exp_dir}/relative_heatmap_{col}_{row}_{across}.png")
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

    create_heatmaps_rel(exp_dir, df, categories=["threshold", "fR", "initial_density"])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide an experiment directory and dimension.")