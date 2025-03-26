import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wasserstein_distance
import seaborn as sns
from sklearn.preprocessing import scale

from classification.common import df_to_xy, read_and_clean_features
from common import get_data_path


def color_by_statistic(features, split_char=" "):
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


def plot_feature_selection(save_loc, measurement, condition, df):
    df["Feature"] = df["Feature"].str.replace("_", " ")
    df = df.sort_values(measurement, ascending=False)
    df["Statistic"] = df["Feature"].map(color_by_statistic(df["Feature"].unique()))

    if condition is None:
        file_name = f"{measurement}"
        title = f"Feature {measurement}"
    else:
        file_name = f"{measurement}_{condition}"
        title = f"Feature {measurement}\n{condition}"

    fig, ax = plt.subplots(figsize=(6, 10))
    sns.barplot(data=df, x=measurement, y="Feature", ax=ax,
                hue="Statistic", palette=sns.color_palette("Set2"),
                hue_order=sorted(df["Statistic"].unique()))
    ax.set(title=title)
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")
    plt.close()


def pairwise_distributions(X_i, X_j, feature_names, condition):
    X_i = list(zip(*X_i))
    X_j = list(zip(*X_j))
    data = []
    for k,name in enumerate(feature_names):
        feature_i = X_i[k]
        feature_j = X_j[k]
        wass = wasserstein_distance(feature_i, feature_j)
        data.append({"Feature": name,
                     "Wasserstein Distance":wass,
                     "Pair": condition})
    return data


def run_pairwise_distributions(save_loc, X, y, int_to_class, feature_names, topn=None):
    for i in range(len(int_to_class)):
        i_indices = [k for k in range(len(y)) if y[k] == i]
        i_X_pair = [X[k] for k in i_indices]
        for j in range(i+1, len(int_to_class)):
            j_indices = [k for k in range(len(y)) if y[k] == j]
            j_X_pair = [X[k] for k in j_indices]
            pair_name = f"{int_to_class[i]} - {int_to_class[j]}"
            if len(i_indices) == 0 or len(j_indices) == 0:
                continue
            data = pairwise_distributions(i_X_pair, j_X_pair, feature_names, pair_name)
            df = pd.DataFrame(data)
            plot_feature_selection(save_loc, "Wasserstein Distance", pair_name, df, topn)
            print(df.nlargest(topn, "Wasserstein Distance"))
            print(df["Wasserstein Distance"].quantile([0.25, 0.5, 0.75]))
            print()


def main(experiment_name, data_types):
    label = ["game"]
    parent_dir = "."
    if len(data_types) == 1:
        parent_dir = data_types[0]
    save_loc = get_data_path(parent_dir, f"model/{experiment_name}/pairwise_games")
    feature_df = read_and_clean_features(data_types, label, experiment_name)
    X, y, int_to_class, feature_names = df_to_xy(feature_df, label[0])
    X = scale(X, axis=0)
    run_pairwise_distributions(save_loc, X, y, int_to_class, feature_names)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide a feature set and the data types to train the model with.")
