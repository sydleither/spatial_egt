"""Visualize the coordinates of a list of samples.

Expected usage: python3 -m spatial_egt.data_analysis.plot_spatial data_type time source sample_ids

Where:
data_type: the name of the directory in data/ containing the processed/ data
time: timepoint
source: the name of the source of the data
sample_ids: a list of the sample_ids to visualize
"""

import sys

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatial_egt.common import game_colors, get_data_path


def main(data_type, time, source, *sample_ids):
    processed_data_path = get_data_path(data_type, "processed", time)
    image_data_path = get_data_path(data_type, "images", time)
    fig, ax = plt.subplots(1, len(sample_ids), figsize=(5*len(sample_ids), 5))
    if len(sample_ids) == 1:
        ax = [ax]
    for s,sample_id in enumerate(sorted(sample_ids)):
        file_name = f"{source} {sample_id}.csv"
        df = pd.read_csv(f"{processed_data_path}/{file_name}")
        df["color"] = df["type"].map({"sensitive":1, "resistant":2})
        grid = np.zeros((df["y"].max()+1, df["x"].max()+1), dtype=int)
        for x, y, color in df[["x", "y", "color"]].values:
            grid[y, x] = color
        colors = ListedColormap(
            ["#000000", game_colors["Sensitive Wins"], game_colors["Resistant Wins"]]
        )
        ax[s].imshow(grid, cmap=colors, vmin=0, vmax=len(colors.colors)-1)
        ax[s].get_xaxis().set_visible(False)
        ax[s].get_yaxis().set_visible(False)
        ax[s].set_title(sample_id)
    fig.suptitle(source)
    fig.subplots_adjust(wspace=0.04, hspace=0)
    fig.figure.patch.set_alpha(0.0)
    save_name = source + "_" + "_".join(sample_ids)
    plt.savefig(f"{image_data_path}/{save_name}.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    if len(sys.argv) > 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3], *sys.argv[4:])
    else:
        print("Please provide the data type, time, source, and sample ids to plot.")
