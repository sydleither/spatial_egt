import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

from common import read_all, plot_line
from extinction import get_extinction_times


def neighborhood_size(df, exp_dir, group1_type, group2_type, group3_type, sim_type):
    colors = ["sienna", "green", "steelblue"]
    for group3 in sorted(df[group3_type].unique()):
        figure, axis = plt.subplots(1, 1, figsize=(8,5), dpi=150)
        for i,group1 in enumerate(sorted(df[group1_type].unique())):
            df_ets = []
            for group2 in sorted(df[group2_type].unique()):
                df_et_i = pd.DataFrame()
                df_cond = df.loc[(df[group1_type] == group1) & (df[group2_type] == group2) & (df[group3_type] == group3)]
                df_et_i["extinction_time"] = get_extinction_times(df_cond, sim_type)
                df_et_i[group1_type] = group1
                df_et_i[group2_type] = group2
                df_ets.append(df_et_i)
            df_et_1 = pd.concat(df_ets)
            df_et_1["rep"] = df_et_1.index
            #axis.scatter(df_et_1[group2_type], df_et_1["extinction_time"], label=group1, color=colors[i])
            plot_line(axis, df_et_1, group2_type, "extinction_time", colors[i], group1)
        axis.legend(framealpha=0.33, title=group1_type)
        axis.set(xlabel="Number of Neighbors in Hood", ylabel="Time of Extinction")
        figure.tight_layout(rect=[0, 0.03, 1, 0.95])
        figure.suptitle(f"Time of Extinction of Either Cell Line in {sim_type} {group3} Experiments")
        plt.savefig(f"output/{exp_dir}/{group3}_{group1_type}_{sim_type}_extinction_times.png", transparent=False)
        plt.close()


def main(exp_dir, sim_type):
    hood_sizes_2D = {1:4, 2:12, 3:28, 4:48, 5:80}
    hood_sizes_3D = {1:6, 2:32, 3:122, 4:256, 5:514}
    df = read_all(exp_dir)
    df["hood"] = pd.to_numeric(df["condition"].str[-1])
    df["game_space"] = df["condition"].str[0:-1]
    df_2d = df.loc[df["dimension"] == "2D"].reset_index()
    df_2d["num_neighbors"] = df_2d["hood"].map(hood_sizes_2D)
    df_3d = df.loc[df["dimension"] == "3D"].reset_index()
    df_3d["num_neighbors"] = df_3d["hood"].map(hood_sizes_3D)
    df = pd.concat([df_2d, df_3d])
    
    neighborhood_size(df, exp_dir, "dimension", "num_neighbors", "game_space", sim_type)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])