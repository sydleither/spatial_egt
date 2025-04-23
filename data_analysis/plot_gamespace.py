import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from spatial_egt.common import get_data_path


def plot_gamespace(save_loc, save_name, df, hue):
    df["C - A"] = df["c"] - df["a"]
    df["B - D"] = df["b"] - df["d"]
    palette = sns.color_palette("hls", len(df[hue].unique()))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.scatterplot(data=df, x="C - A", y="B - D", s=100, hue=hue, ax=ax, palette=palette)
    ax.axvline(0, color="black")
    ax.axhline(0, color="black")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    plt.savefig(f"{save_loc}/gamespace_{save_name}.png")


def get_samples(data_type, source, sample_ids):
    data_path = get_data_path(data_type, ".")
    df_labels = pd.read_csv(f"{data_path}/labels.csv")
    df_labels["sample"] = df_labels["sample"].astype(str)
    if source != "":
        df_labels = df_labels[df_labels["source"] == source]
    if sample_ids:
        df_labels = df_labels[df_labels["sample"].isin(sample_ids)]
    return df_labels


def main(data_type, hue, *filter_args):
    save_name = hue
    source = ""
    sample_ids = None
    if len(filter_args) == 1:
        source = filter_args[0]
        save_name += "_" + source
    if len(filter_args) > 1:
        source = filter_args[0]
        sample_ids = filter_args[1:]
        save_name += "_" + source + "_" + "_".join(sample_ids)

    image_data_path = get_data_path(data_type, "images")
    df = get_samples(data_type, source, sample_ids)
    plot_gamespace(image_data_path, save_name, df, hue)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(*sys.argv[1:])
    else:
        print(
            "Please provide the data type, hue, (optionally) source, and (optionally) sample ids to plot."
        )
