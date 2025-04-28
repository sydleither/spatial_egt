import sys

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatial_egt.common import game_colors, get_data_path


def plot_sample(df, save_loc, sample_id):
    color_map = {"sensitive":1, "resistant":2}
    df["color"] = df["type"].map(color_map)
    max_x = df["x"].max() + 1
    max_y = df["y"].max() + 1
    grid = np.zeros((max_y, max_x))
    for _, row in df.iterrows():
        grid[row["y"], row["x"]] = row["color"]

    scale = 4
    scaled_grid = np.kron(grid, np.ones((scale, scale)))
    fig, ax = plt.subplots(figsize=(scale*max_x, scale*max_y), dpi=1)
    cell_colors = [game_colors["Sensitive Wins"],
                   game_colors["Resistant Wins"]]
    cmap = ListedColormap(["#F0F0F0"]+cell_colors)
    plt.imshow(scaled_grid, cmap=cmap, interpolation="none")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    fig.tight_layout()
    plt.savefig(f"{save_loc}/sample_{sample_id}.png", bbox_inches="tight")


def main(data_type, source, *sample_ids):
    processed_data_path = get_data_path(data_type, "processed")
    image_data_path = get_data_path(data_type, "images")
    for sample_id in sample_ids:
        file_name = f"{source} {sample_id}.csv"
        df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
        plot_sample(df_sample, image_data_path, sample_id)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], *sys.argv[3:])
    else:
        print("Please provide the data type, source, and sample ids to plot.")
