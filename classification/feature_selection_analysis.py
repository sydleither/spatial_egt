import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import get_data_path, theme_colors
from classification.feature_pairwise_games import plot_feature_selection


def feature_set_plot(data_path, feature_set_size, df, xlabel, n=10):
    df_top = df.nlargest(n, xlabel)
    fig, ax = plt.subplots(figsize=(6, (n*feature_set_size)//3))
    sns.barplot(data=df_top, x=xlabel, y="features", color=theme_colors[0], ax=ax)
    ax.set(title=feature_set_size, xlabel=xlabel, ylabel="Feature")
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{data_path}/{feature_set_size}.png", bbox_inches="tight")
    plt.close()


def main(data_type, data_source, feature_names):
    feature_dir = "_".join(feature_names)
    data_path = get_data_path(data_type, f"model/{feature_dir}/{data_source}")
    df = pd.DataFrame()
    for file in os.listdir(data_path):
        if not file.endswith(".csv"):
            continue
        df_fs = pd.read_csv(f"{data_path}/{file}")
        cols = [x for x in df_fs.columns if x != "value"]
        df_fs["features"] = df_fs[cols].apply(lambda row: "\n".join(row.values.astype(str)), axis=1)
        df_fs["feature_set_size"] = int(file[:-4])
        df = pd.concat([df, df_fs[["feature_set_size", "features", "value"]]])

    if data_source == "fragmentation":
        xlabel = "Entropy Shared with Game"
    else:
        xlabel = "Mean Accuracy"
    df[xlabel] = df["value"]
    df = df.drop(["value"], axis=1)

    for feature_set_size in df["feature_set_size"].unique():
        df_num = df[df["feature_set_size"] == feature_set_size]
        if feature_set_size == 1:
            df_num["Feature"] = df_num["features"]
            plot_feature_selection(data_path, xlabel, None, df_num)
        else:
            feature_set_plot(data_path, feature_set_size, df_num, xlabel)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        print("Please provide the data type, data source (fragmentation or model_iteration), and feature set/names.")
