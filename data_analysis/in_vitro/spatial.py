import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import (cell_colors, get_experiment_names,
                    make_image_dir, processed_data_path)


def calculate_fs(df, key):
    df_count = df.groupby(key+["type"]).count().reset_index()
    df_sum = df_count.groupby(key)["x"].sum()
    df_sum = df_sum.reset_index()
    df_sum = df_sum.rename(columns={"x":"TotalCells"})
    df_sens = df_count[df_count["type"] == "sensitive"]
    df_sens = df_sum.merge(df_sens, on=key)
    df_sens["fs"] = df_sens["x"]/df_sens["TotalCells"]
    df_sens = df_sens[key+["fs"]]
    df = df.merge(df_sens, on=key)
    return df


def plot_plate_fs(df, exp_name):
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
    plt.savefig(f"data/in_vitro/images/{exp_name}/plate_fs.png")


def plot_plate_sections(df, exp_name):
    for drugcon in df["DrugConcentration"].unique():
        df_dc = df[df["DrugConcentration"] == drugcon]
        well_letters = sorted(df_dc["WellId"].str[0].unique())
        well_nums = sorted(df_dc["WellId"].str[1:].astype(int).unique())
        num_letters = len(well_letters)
        num_nums = len(well_nums)
        fig, ax = plt.subplots(num_letters, num_nums, figsize=(10*num_nums, 10*num_letters))
        for l in range(len(well_letters)):
            for n in range(len(well_nums)):
                well = well_letters[l]+str(well_nums[n])
                sns.scatterplot(data=df[df["WellId"] == well], x="x", y="y", 
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
        plt.savefig(f"data/in_vitro/images/{exp_name}/plate_{drugcon_str}uM.png")


def plot_single_well(df, exp_name, well):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df[df["WellId"] == well], x="x", y="y", 
                    hue="type", legend=False, ax=ax,
                    palette=cell_colors, 
                    hue_order=["sensitive", "resistant"])
    ax.set(xlabel="", ylabel="")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_facecolor("lightgrey")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    plt.savefig(f"data/in_vitro/images/{exp_name}/well_{well}.png")


def main():
    for exp_name in get_experiment_names():
        make_image_dir(exp_name)
        df = pd.read_csv(f"{processed_data_path}/spatial_{exp_name}.csv")
        plot_plate_sections(df, exp_name)
        plot_plate_fs(df, exp_name)
        plot_single_well(df, exp_name, "F5")


if __name__ == "__main__":
    main()
