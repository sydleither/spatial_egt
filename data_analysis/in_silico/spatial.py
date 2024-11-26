import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import game_colors, get_data_path
from data_processing.spatial_statistics import calculate_game

cell_colors = [game_colors["sensitive_wins"], game_colors["resistant_wins"]]


def plot_games(df, save_loc):
    fig, ax = plt.subplots(1, 4, figsize=(32, 8))
    for a,game in enumerate(game_colors):
        df_game = df[df["game"] == game]
        sns.scatterplot(data=df_game, x="x", y="y", 
                        hue="type", legend=False, ax=ax[a],
                        palette=cell_colors, size=1, markers="s",
                        hue_order=["sensitive", "resistant"],
                        edgecolors="none")
        ax[a].set(title=game)
        ax[a].set(xlabel="", ylabel="")
        ax[a].set(xlim=(0,125), ylim=(0,125))
        ax[a].get_xaxis().set_ticks([])
        ax[a].get_yaxis().set_ticks([])
        ax[a].set_facecolor("whitesmoke")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    plt.savefig(f"{save_loc}/visualization.png")


def main():
    processed_data_path = get_data_path("in_silico", "processed")
    image_data_path = get_data_path("in_silico", "images")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["game"] = df_payoff.apply(calculate_game, axis="columns")
    # df_payoff = df_payoff[(df_payoff["initial_fr"] >= 0.45) & (df_payoff["initial_fr"] <= 0.55)]
    # df_payoff = df_payoff[(df_payoff["initial_density"] >= 6800) & (df_payoff["initial_density"] <= 8800)]
    df = pd.DataFrame()
    sample_ids = dict()
    for game in game_colors:
        game_samples = df_payoff[df_payoff["game"] == game]
        sample_id = random.sample(list(game_samples["sample"].values), 1)[0]
        file_name = f"spatial_HAL_{sample_id}.csv"
        df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
        sample_ids[game] = int(sample_id)
        df_sample["game"] = game
        df = pd.concat([df, df_sample])
    plot_games(df, image_data_path)
    print(sample_ids)


if __name__ == "__main__":
    main()
