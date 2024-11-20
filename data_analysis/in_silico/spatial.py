import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import game_colors, in_vitro_exp_names, get_data_path
from data_processing.spatial_statistics import calculate_game

cell_colors = [game_colors["sensitive_wins"], game_colors["resistant_wins"]]


def plot_plate_sections(df, save_loc, exp_name):
    for drugcon in df["DrugConcentration"].unique():
        df_dc = df[df["DrugConcentration"] == drugcon]
        well_letters = sorted(df_dc["sample"].str[0].unique())
        well_nums = sorted(df_dc["sample"].str[1:].astype(int).unique())
        num_letters = len(well_letters)
        num_nums = len(well_nums)
        fig, ax = plt.subplots(num_letters, num_nums, figsize=(10*num_nums, 10*num_letters))
        for l in range(len(well_letters)):
            for n in range(len(well_nums)):
                well = well_letters[l]+str(well_nums[n])
                sns.scatterplot(data=df[df["sample"] == well], x="x", y="y", 
                                hue="type", legend=False, ax=ax[l][n],
                                palette=cell_colors, 
                                hue_order=["sensitive", "resistant"])
                ax[l][n].set(xlabel="", ylabel="")
                ax[l][n].get_xaxis().set_ticks([])
                ax[l][n].get_yaxis().set_ticks([])
                ax[l][n].set_facecolor("lightgrey")
        fig.patch.set_alpha(0.0)
        fig.tight_layout()
        drugcon_str = "{:10.3f}".format(drugcon).strip().replace(".", "")
        plt.savefig(f"{save_loc}/{exp_name}/plate_{drugcon_str}uM.png")


def plot_single_well(df, save_loc, exp_name, well):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df[df["sample"] == well], x="x", y="y", 
                    hue="type", legend=False, ax=ax,
                    palette=cell_colors, 
                    hue_order=["sensitive", "resistant"])
    ax.set(xlabel="", ylabel="")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_facecolor("lightgrey")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    plt.savefig(f"{save_loc}/{exp_name}/well_{well}.png")


def main():
    processed_data_path = get_data_path("in_silico", "processed")
    image_data_path = get_data_path("in_silico", "images")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["game"] = df_payoff.apply(calculate_game, axis="columns")
    for game in game_colors:
        game_samples = df_payoff[df_payoff["game"] == game]
        for sample in game_samples["sample"]:
            file_name = f"spatial_HAL_{sample}.csv"
            df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
    # df = pd.DataFrame()
    # for file_name in os.listdir(processed_data_path):
    #     if not file_name.startswith("spatial"):
    #         continue
    #     source = file_name.split("_")[1]
    #     sample = file_name.split("_")[2][:-4]
    #     df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
    #     df_sample["source"] = source
    #     df_sample["sample"] = sample
    #     df = pd.concat([df, df_sample])
    # plot_plate_sections(df, image_data_path, exp_name)
    # plot_single_well(df, image_data_path, exp_name, "F5")


if __name__ == "__main__":
    main()
