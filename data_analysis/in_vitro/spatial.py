import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import game_colors, get_data_path

cell_colors = [game_colors["sensitive_wins"], game_colors["resistant_wins"]]


def plot_plate_sections(df, save_loc, exp_name):
    df["well_letter"] = df["sample"].str[0]
    df["well_num"] = df["sample"].str[1:].astype(int)
    for drugcon in df["DrugConcentration"].unique():
        df_dc = df[df["DrugConcentration"] == drugcon]
        facet = sns.FacetGrid(df_dc, col="well_num", row="well_letter",
                              row_order=sorted(df_dc["well_letter"].unique()),
                              col_order=sorted(df_dc["well_num"].unique()), 
                              height=8, aspect=1)
        facet.map_dataframe(sns.scatterplot, x="x", y="y",hue="type", legend=False,
                            palette=cell_colors, size=1, edgecolors="none",
                            hue_order=["sensitive", "resistant"])
        facet.set_titles(col_template="{col_name}", row_template="{row_name}")
        facet.set(facecolor="whitesmoke")
        facet.tight_layout()
        drugcon_str = "{:10.3f}".format(drugcon).strip().replace(".", "")
        facet.savefig(f"{save_loc}/{exp_name}/plate_{drugcon_str}uM.png", bbox_inches="tight")


def plot_single_well(df, save_loc, exp_name, well):
    fig, ax = plt.subplots(figsize=(10,10))
    sns.scatterplot(data=df[df["sample"] == well], x="x", y="y", 
                    hue="type", legend=False, ax=ax,
                    palette=cell_colors, size=1, edgecolors="none",
                    hue_order=["sensitive", "resistant"])
    ax.set(xlabel="", ylabel="")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_facecolor("whitesmoke")
    fig.tight_layout()
    plt.box(False)
    plt.savefig(f"{save_loc}/{exp_name}/well_{well}.png", bbox_inches="tight")


def main():
    processed_data_path = get_data_path("in_vitro", "processed")
    image_data_path = get_data_path("in_vitro", "images")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    for exp_name in df_payoff["source"].unique():
        df = pd.DataFrame()
        exp_samples = df_payoff[df_payoff["source"] == exp_name]
        for well in exp_samples["sample"].values:
            file_name = f"spatial_{exp_name}_{well}.csv"
            df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
            df_sample["sample"] = well
            drugcon = exp_samples[exp_samples["sample"] == well]["DrugConcentration"].iloc[0]
            df_sample["DrugConcentration"] = drugcon
            df = pd.concat([df, df_sample])
        plot_plate_sections(df, image_data_path, exp_name)
        plot_single_well(df, image_data_path, exp_name, "F5")


if __name__ == "__main__":
    main()
