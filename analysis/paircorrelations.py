import os
import sys
import warnings
warnings.filterwarnings("ignore")

from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd


def read_dim(exp_dir, exp_name, dimension):
    df = pd.DataFrame()
    run_path = f"output/{exp_dir}/{exp_name}"
    for rep_dir in os.listdir(run_path):
        rep_path = f"{run_path}/{rep_dir}"
        if os.path.isfile(rep_path):
            continue
        result_file = f"{rep_path}/{dimension}pairCorrelations.csv"
        if not os.path.exists(result_file) or os.path.getsize(result_file) == 0:
            print(f"File not found for rep {rep_dir}")
            continue
        df_i = pd.read_csv(result_file)
        df_i["rep"] = int(rep_dir)
        df_i["condition"] = exp_name
        df_i["dimension"] = result_file[0:2]
        df = pd.concat([df, df_i])
    return df


def main(exp_dir, exp_name, dimension, transparent=False):
    df = read_dim(exp_dir, exp_name, dimension)

    colors = cm.get_cmap("Set3").colors
    for pair in df["pair"].unique():
        df_pair = df.loc[df["pair"] == pair]
        for time in df_pair["time"].unique():
            df_time = df_pair.loc[df_pair["time"] == time]
            fig, ax = plt.subplots()
            for rep in df_time["rep"].unique():
                df_rep = df_time.loc[df_time["rep"] == rep]
                ax.bar(df_rep["distance"], df_rep["count"], alpha=0.6, color=colors[rep])
            ax.set_xlabel("Distance Between Pairs of Cells")
            ax.set_ylabel("Count of Pairs")
            ax.set_title(f"Pair Correlation for {dimension} {exp_name} {pair} pairs at time {time}")
            fig.tight_layout()
            if transparent:
                fig.patch.set_alpha(0.0)
            fig.savefig(f"output/{exp_dir}/{exp_name}/{dimension}pair_correlation_{pair}_{time}.png")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide an experiment directory, name, and dimension.")