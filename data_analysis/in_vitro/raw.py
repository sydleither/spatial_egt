import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common.common import (game_colors, cell_type_map, 
                    in_vitro_exp_names, get_data_path)

pd.set_option('mode.chained_assignment', None)
cell_colors = [game_colors["sensitive_wins"], game_colors["resistant_wins"]]


def plot_growth_over_time(df, save_loc, exp_name):
    well_letters = sorted(df["WellId"].str[0].unique())
    well_nums = sorted(df["WellId"].str[1:].astype(int).unique())
    num_letters = len(well_letters)
    num_nums = len(well_nums)
    fig, ax = plt.subplots(num_letters, num_nums, 
                           figsize=(3*num_nums, 3*num_letters),
                           sharex=True, sharey=True)
    for l in range(len(well_letters)):
        for n in range(len(well_nums)):
            well = well_letters[l]+str(well_nums[n])
            sns.lineplot(data=df[df["WellId"] == well], x="Time", y="Count", 
                         hue="CellType", legend=False, ax=ax[l][n],
                         palette=cell_colors, linewidth=5,
                         hue_order=["sensitive", "resistant"])
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    plt.savefig(f"{save_loc}/{exp_name}/plate_gr.png")


def calculate_fs(df, key):
    df_count = df.groupby(key+["CellType"]).sum().reset_index()
    df_sum = df_count.groupby(key)["Count"].sum()
    df_sum = df_sum.reset_index()
    df_sum = df_sum.rename(columns={"Count":"TotalCells"})
    df_sens = df_count[df_count["CellType"] == "sensitive"]
    df_sens = df_sum.merge(df_sens, on=key)
    df_sens["fs"] = df_sens["Count"]/df_sens["TotalCells"]
    df_sens = df_sens[key+["fs"]]
    df = df.merge(df_sens, on=key)
    return df


def plot_plate_fs(df, save_loc, exp_name, fs_time):
    key = ["WellId", "PlateId"]
    df_hm = calculate_fs(df, key)
    df_hm["WellLetter"] = df_hm["WellId"].str[0]
    df_hm["WellNum"] = df_hm["WellId"].str[1:].astype(int)
    df_hm = df_hm[["WellLetter", "WellNum", "fs"]].drop_duplicates()
    df_hm = df_hm.pivot(index="WellLetter", columns="WellNum", values="fs")
    fig, ax = plt.subplots()
    ax = sns.heatmap(df_hm, annot=True, fmt=".2f", linewidth=1, ax=ax)
    ax.set(title="Fraction Sensitive")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    plt.savefig(f"{save_loc}/{exp_name}/plate_fs_{fs_time}.png")


def plot_game_gr(df, save_loc, exp_name):
    sns.lmplot(data=df, x="Fraction_Sensitive", y="GrowthRate", 
               hue="CellType", col="DrugConcentration", legend=False,
               palette=cell_colors, hue_order=["sensitive", "resistant"],
               facet_kws=dict(sharey=False))
    plt.savefig(f"{save_loc}/{exp_name}/gr_by_fs.png", transparent=True)


def main():
    raw_data_path = get_data_path("in_vitro", "raw")
    image_data_path = get_data_path("in_vitro", "images")

    for exp_name in in_vitro_exp_names:
        counts_name = f"counts_df_processed_{exp_name}_plate1.csv"
        df = pd.read_csv(f"{raw_data_path}/{counts_name}")
        df["CellType"] = df["CellType"].map(cell_type_map)
        plot_growth_over_time(df, image_data_path, exp_name)
        plot_plate_fs(df[df["Time"] == df["Time"].max()], 
                      image_data_path, exp_name, "end")
        plot_plate_fs(df[df["Time"] == df["Time"].min()], 
                      image_data_path, exp_name, "start")

        growth_name = f"growth_rate_df_{exp_name}_plate1.csv"
        df = pd.read_csv(f"{raw_data_path}/{growth_name}")
        df["CellType"] = df["CellType"].map(cell_type_map)
        plot_game_gr(df, image_data_path, exp_name)


if __name__ == "__main__":
    main()