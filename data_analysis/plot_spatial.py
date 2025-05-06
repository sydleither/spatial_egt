"""Visualize the coordinates of a list of samples.

Expected usage: python3 -m spatial_egt.data_analysis.plot_spatial data_type source sample_ids

Where:
data_type: the name of the directory in data/ containing the processed/ data
source: the name of the source of the data
sample_ids: a list of the sample_ids to visualize
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

from spatial_egt.common import game_colors, get_data_path


def main(data_type, source, *sample_ids):
    processed_data_path = get_data_path(data_type, "processed")
    image_data_path = get_data_path(data_type, "images")
    fig, ax = plt.subplots(1, len(sample_ids), figsize=(5*len(sample_ids), 5))
    if len(sample_ids) == 1:
        ax = [ax]
    for s,sample_id in enumerate(sorted(sample_ids)):
        file_name = f"{source} {sample_id}.csv"
        df = pd.read_csv(f"{processed_data_path}/{file_name}")
        df["color"] = df["type"].map(
            {"sensitive":game_colors["Sensitive Wins"], "resistant":game_colors["Resistant Wins"]}
        )
        ax[s].scatter(x=df["x"], y=df["y"], s=2, c=df["color"])
        ax[s].set_title(sample_id)
    fig.suptitle(source)
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    save_name = source + "_" + "_".join(sample_ids)
    plt.savefig(f"{image_data_path}/{save_name}.png", bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], *sys.argv[3:])
    else:
        print("Please provide the data type, source, and sample ids to plot.")
