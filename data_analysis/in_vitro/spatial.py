import sys

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
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
                              height=6, aspect=1)
        facet.map_dataframe(sns.scatterplot, x="x", y="y", hue="type", legend=False,
                            palette=cell_colors, size=1, edgecolors="none",
                            hue_order=["sensitive", "resistant"])
        facet.set_titles(col_template="{col_name}", row_template="{row_name}")
        facet.set(facecolor="whitesmoke")
        facet.tight_layout()
        facet.figure.patch.set_alpha(0.0)
        drugcon_str = "{:10.3f}".format(drugcon).strip().replace(".", "")
        facet.savefig(f"{save_loc}/{exp_name}/plate_{drugcon_str}uM.png", bbox_inches="tight")


def plot_single_well(df, save_loc, exp_name, well):
    color_map = {"sensitive":1, "resistant":2}
    df["color"] = df["type"].map(color_map)
    max_x = df["x"].max() + 1
    max_y = df["y"].max() + 1
    grid = np.zeros((max_y, max_x))
    for _, row in df.iterrows():
        grid[row["y"], row["x"]] = row["color"]

    fig, ax = plt.subplots(figsize=(max_x, max_y), dpi=1)
    cmap = ListedColormap(["#F0F0F0"]+cell_colors)
    plt.imshow(grid, cmap=cmap, interpolation="none")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    fig.tight_layout()
    plt.savefig(f"{save_loc}/{exp_name}/well_{well}.png", bbox_inches="tight")


#https://stackoverflow.com/questions/18666014/downsample-array-in-python
def downsample(ar, fact):
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy//fact * (X//fact) + Y//fact
    res = ndimage.maximum(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx//fact, sy//fact)
    return res


def plot_single_well_downsampled(df, save_loc, exp_name, well):
    color_map = {"sensitive":1, "resistant":2}
    df["color"] = df["type"].map(color_map)
    grid = np.zeros((1250, 1250))
    for _, row in df.iterrows():
        grid[row["y"], row["x"]] = row["color"]
    grid = downsample(grid, 10)

    scale = 4
    fig, ax = plt.subplots(figsize=(scale*grid.shape[0], scale*grid.shape[1]), dpi=1)
    cmap = ListedColormap(["#F0F0F0"]+cell_colors)
    plt.imshow(grid, cmap=cmap, interpolation="none")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    fig.tight_layout()
    plt.savefig(f"{save_loc}/{exp_name}/well_{well}_ds.png", bbox_inches="tight")


def plate_main(source):
    processed_data_path = get_data_path("in_vitro", "processed")
    image_data_path = get_data_path("in_vitro", "images")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    exp_samples = df_payoff[df_payoff["source"] == source]
    df = pd.DataFrame()
    for well in exp_samples["sample"].unique():
        file_name = f"spatial_{source}_{well}.csv"
        df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
        df_sample["sample"] = well
        drugcon = exp_samples[exp_samples["sample"] == well]["DrugConcentration"].iloc[0]
        df_sample["DrugConcentration"] = drugcon
        df = pd.concat([df, df_sample])
    plot_plate_sections(df, image_data_path, source)


def well_main(source, *wells):
    processed_data_path = get_data_path("in_vitro", "processed")
    image_data_path = get_data_path("in_vitro", "images")
    for well in wells:
        file_name = f"spatial_{source}_{well}.csv"
        df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
        plot_single_well(df_sample, image_data_path, source, well)
        plot_single_well_downsampled(df_sample, image_data_path, source, well)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        well_main(sys.argv[1], *sys.argv[2:])
    elif len(sys.argv) == 2:
        plate_main(sys.argv[1])
    else:
        print("Please provide a source and, optionally, the wells to visualize.")
