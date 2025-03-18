import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import get_data_path


def plot_gamespace(save_loc, save_name, df, hue):
    df["C - A"] = df["c"] - df["a"]
    df["B - D"] = df["b"] - df["d"]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.scatterplot(data=df, x="C - A", y="B - D",
                    hue=hue, palette="Set2", ax=ax)
    ax.axvline(0, color="black")
    ax.axhline(0, color="black")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    plt.savefig(f"{save_loc}/gamespace_{save_name}.png")


def get_samples(data_type, source, sample_ids): #TODO combine with plot_function.get_data
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    df_payoff = df_payoff[df_payoff["game"] != "Unknown"]
    if not source == "":
        df_payoff = df_payoff[df_payoff["source"] == source]
    if sample_ids:
        df_payoff = df_payoff[df_payoff["sample"].isin(sample_ids)]
    return df_payoff


def main(data_type, source, sample_ids):
    image_data_path = get_data_path(data_type, "images")
    df = get_samples(data_type, source, sample_ids)
    hue = "source"
    save_name = "all"
    if not source == "":
        save_name = source
    if sample_ids:
        hue = "sample"
        save_name = source + "_" + "_".join(sample_ids)
    plot_gamespace(image_data_path, save_name, df, hue)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1], "", None)
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2], None)
    elif len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        print("Please provide the data type, (optionally) source, and (optionally) sample ids to plot.")
