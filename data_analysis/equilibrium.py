import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import game_colors, get_data_path
from data_analysis.plot_gamespace import get_samples


def main(data_type, source, sample_ids):
    features_data_path = get_data_path(data_type, "features")
    df_features = pd.read_csv(f"{features_data_path}/all.csv")
    df_features["sample"] = df_features["sample"].astype(str)
    df = get_samples(data_type, source, sample_ids)
    df["Stationary Solution"] = (df["c"]-df["a"])/((df["c"]-df["a"])+(df["b"]-df["d"]))
    df.loc[df["game"] == "Sensitive Wins", "Stationary Solution"] = 0
    df.loc[df["game"] == "Resistant Wins", "Stationary Solution"] = 1
    df = df.merge(df_features, on=["source", "sample", "game"])
    df["Proportion Resistant"] = 1 - df["Proportion_Sensitive"]

    image_data_path = get_data_path(data_type, "images")
    sns.lmplot(data=df, x="Stationary Solution", y="Proportion Resistant",
               hue="game", hue_order=game_colors.keys(), palette=game_colors.values())
    # sns.lmplot(data=df[df["game"] == "Coexistence"], x="SFP_Skew", y="SFP_Mean")
    plt.savefig(f"{image_data_path}/stationary_solution.png", transparent=True)
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1], "", None)
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2], None)
    elif len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        print("Please provide the data type, (optionally) source, and (optionally) sample ids to plot.")
