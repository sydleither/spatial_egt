import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common.common import game_colors, get_data_path
from data_processing.spatial_statistics import calculate_game


def plot_feature(df, save_loc, feature_name):
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=feature_name, hue="game",
                 hue_order=game_colors.keys(),
                 palette=game_colors.values(),
                 bins=10, multiple="stack", ax=ax)
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/sample_{feature_name}.png")


def main():
    processed_data_path = get_data_path("in_silico", "processed")
    save_loc = get_data_path("in_silico", "images")
    df = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df["sample"] = df["sample"].astype(str)
    df["game"] = df.apply(calculate_game, axis="columns")
    plot_feature(df, save_loc, "initial_density")
    plot_feature(df, save_loc, "initial_fr")


if __name__ == "__main__":
    main()