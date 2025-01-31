import matplotlib.pyplot as plt
import seaborn as sns

from common import game_colors, get_data_path, read_payoff_df


game_colors["unknown"] = "gray"


def plot_feature(df, save_loc, feature_name):
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=feature_name, hue="game",
                 hue_order=game_colors.keys(),
                 palette=game_colors.values(),
                 bins=11, multiple="stack", ax=ax)
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/sample_{feature_name}.png")


def main():
    processed_data_path = get_data_path("in_silico", "processed")
    save_loc = get_data_path("in_silico", "images")
    df = read_payoff_df(processed_data_path)
    plot_feature(df, save_loc, "initial_density")
    plot_feature(df, save_loc, "initial_fr")


if __name__ == "__main__":
    main()