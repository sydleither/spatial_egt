import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wasserstein_distance
import seaborn as sns
from sklearn.preprocessing import scale

from classification.common import df_to_xy, get_feature_data


def label_statistic(features, split_char=" "):
    extra = [
        "Sensitive", "Resistant",
        "Local", "Global",
        "Mean", "SD", "Skew", "Kurtosis",
        "Min", "Max", "0"
    ]
    feature_categories = []
    feature_to_statistic = dict()
    for feature in features:
        feature_category = [x for x in feature.split(split_char) if x not in extra]
        feature_category = split_char.join(feature_category)
        if feature_category == "Proportion":
            feature_category = "Proportion"+split_char+"Sensitive"
        feature_to_statistic[feature] = feature_category
        if feature_category not in feature_categories:
            feature_categories.append(feature_category)
    return feature_to_statistic


def format_df(df):
    df["Feature"] = df["Feature"].str.replace("_", " ")
    df["Statistic"] = df["Feature"].map(label_statistic(df["Feature"].unique()))
    return df


def plot_feature_selection(save_loc, measurement, condition, df):
    df = df.sort_values(measurement, ascending=False)
    statistics = df["Statistic"].unique()

    if condition is None:
        file_name = f"{measurement}"
        title = f"Feature {measurement}"
    else:
        file_name = f"{measurement}_{condition}"
        title = f"Feature {measurement}\n{condition}"

    fig, ax = plt.subplots(figsize=(6, len(df["Feature"].unique())//2))
    sns.barplot(
        data=df, x=measurement, y="Feature", ax=ax,
        palette=sns.color_palette("hls", len(statistics)),
        hue="Statistic", hue_order=sorted(statistics)
    )
    ax.set(title=title)
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")
    plt.close()


def label_bars(ax, labels):
    bars = ax.patches
    sorted_bars = sorted(zip(bars, [b.get_y() for b in bars]), key=lambda x: x[1])
    sorted_bars = list(zip(*sorted_bars))[0]
    for b,bar in enumerate(sorted_bars):
        ax.text(
            bar.get_width()/2, bar.get_y()+bar.get_height()/2,
            labels[b], ha="center", va="center",
            fontsize=10, color="black"
        )
    ax.set(yticklabels=[], ylabel="")
    ax.tick_params(left=False)


def plot_feature_gamespace(save_loc, measurement, int_to_class, df):
    num_games = len(int_to_class)
    statistics = df["Statistic"].unique()
    fig, ax = plt.subplots(num_games, num_games, figsize=(3*num_games, 3*num_games))
    for i in range(num_games):
        for j in range(num_games):
            if i == num_games-2 and j == 1:
                stats = sorted(statistics)
                colors = sns.color_palette("hls", len(statistics))
                leg = [mpatches.Patch(color=colors[k], label=s) for k, s in enumerate(stats)]
                ax[i][j].legend(handles=leg, title="Statistic", loc="center")
                ax[i][j].axis("off")
                continue
            elif i == j or j < i+1:
                fig.delaxes(ax[i][j])
                continue
            i_game = int_to_class[i]
            j_game = int_to_class[j]
            df_pair = df[(df["Game i"] == i_game) & (df["Game j"] == j_game)]
            df_pair = df_pair.nlargest(3, measurement)
            sns.barplot(
                data=df_pair, x=measurement, y="Feature", ax=ax[i][j],
                palette=sns.color_palette("hls", len(statistics)),
                hue="Statistic", hue_order=sorted(statistics), legend=False
            )
            top_features = df_pair["Feature"].values
            label_bars(ax[i][j], top_features)
            if i == 0:
                ax[i][j].set_title(j_game, fontsize=12)
            if j == i+1:
                ax[i][j].set_ylabel(i_game, fontsize=12)
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/feature_gamespace.png", bbox_inches="tight")
    plt.close()


def pairwise_distributions(feature_names, X_i, X_j, game_i, game_j):
    X_i = list(zip(*X_i))
    X_j = list(zip(*X_j))
    data = []
    for k,name in enumerate(feature_names):
        feature_i = X_i[k]
        feature_j = X_j[k]
        wass = wasserstein_distance(feature_i, feature_j)
        data.append({
            "Feature": name,
            "Wasserstein Distance": wass,
            "Pair": f"{game_i} - {game_j}",
            "Game i": game_i,
            "Game j": game_j
        })
    return data


def run_pairwise_distributions(X, y, int_to_class, feature_names):
    rows = []
    for i in range(len(int_to_class)):
        i_indices = [k for k in range(len(y)) if y[k] == i]
        i_features = [X[k] for k in i_indices]
        i_game = int_to_class[i]
        for j in range(i+1, len(int_to_class)):
            j_indices = [k for k in range(len(y)) if y[k] == j]
            j_features = [X[k] for k in j_indices]
            j_game = int_to_class[j]
            if len(i_indices) == 0 or len(j_indices) == 0:
                continue
            data = pairwise_distributions(feature_names, i_features, j_features, i_game, j_game)
            rows += data
    return pd.DataFrame(rows)


def main(data_type, feature_names):
    save_loc, df, feature_names, label_name = get_feature_data(data_type, feature_names)
    feature_df = df[feature_names+[label_name]]
    X, y, int_to_class = df_to_xy(feature_df, feature_names, label_name)
    X = scale(X, axis=0)

    df = run_pairwise_distributions(X, y, int_to_class, feature_names)
    measurements = [i for i in df.columns if i not in ["Feature", "Pair", "Game i", "Game j"]]
    df = format_df(df)
    for measurement in measurements:
        for pair in df["Pair"].unique():
            df_pair = df[df["Pair"] == pair]
            plot_feature_selection(save_loc, measurement, pair, df_pair)
        plot_feature_gamespace(save_loc, measurement, int_to_class, df)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide the data type and the feature set/names.")
