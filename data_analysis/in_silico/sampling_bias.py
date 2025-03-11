import matplotlib.pyplot as plt
import seaborn as sns

from common import game_colors, get_data_path, read_payoff_df


game_colors["unknown"] = "gray"


def plot_scatter(df, save_loc):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, hue="game", ax=ax,
                    x="initial_density", y="initial_fr", 
                    hue_order=game_colors.keys(),
                    palette=game_colors.values())
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/sampling_uniformity.png")


def main():
    processed_data_path = get_data_path("in_silico", "processed")
    save_loc = get_data_path("in_silico", "images")
    df = read_payoff_df(processed_data_path)
    plot_scatter(df, save_loc)


if __name__ == "__main__":
    main()