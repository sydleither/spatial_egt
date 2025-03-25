import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (f_classif,
                                       mutual_info_classif)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
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


def plot_feature_selection(save_loc, measurement, condition, df, topn):
    file_name = f"{measurement}_{condition}"
    df["Feature"] = df["Feature"].str.replace("_", " ")
    if topn is None:
        df = df.sort_values(measurement, ascending=False)
        df["Statistic"] = df["Feature"].map(color_by_statistic(df["Feature"].unique()))
        hue = "Statistic"
        hue_order = sorted(df["Statistic"].unique())
        palette = sns.color_palette("Set2")
        color = None
    else:
        file_name += f"_{topn}"
        df = df.nlargest(topn, measurement)
        hue = None
        hue_order = None
        palette = None
        color = "purple"
    fig, ax = plt.subplots(figsize=(6, 12))
    sns.barplot(data=df, x=measurement, y="Feature", color=color,
                hue=hue, palette=palette, hue_order=hue_order, ax=ax)
    ax.set(title=f"Feature {measurement}\n{condition}")
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")
    plt.close()


def plot_pairwise_distances(save_loc, measurement, df):
    df["Feature"] = df["Feature"].str.replace("_", " ")
    df["Statistic"] = df["Feature"].map(color_by_statistic(df["Feature"].unique()))
    df = df.sort_values("Statistic", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 12))
    sns.barplot(data=df, x=measurement, y="Feature", hue="Pair", ax=ax,
                palette=sns.color_palette("Set2"))
    ax.set(title=f"Feature Pairwise {measurement}")
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/{measurement}_pairwise.png", bbox_inches="tight")
    plt.close()


def rf_importance(X, y, n_repeats=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    rf = RandomForestClassifier(max_depth=10).fit(X_train, y_train)
    result = permutation_importance(rf, X_test, y_test, n_repeats=n_repeats)
    return [np.mean(x) for x in result.importances]


def feature_selection(X, y, feature_names):
    m_info = mutual_info_classif(X, y)
    f_stat, _ = f_classif(X, y)
    rf = rf_importance(X, y)
    data = []
    for i,name in enumerate(feature_names):
        data.append({"Feature": name,
                     "Mutual Information":m_info[i],
                     "F-Statistic":f_stat[i],
                     "Mean Decrease in Test Accuracy":rf[i]})
    df = pd.DataFrame(data)
    return df


def run_feature_selection(save_loc, X, y, feature_names, condition, topn=None):
    df = feature_selection(X, y, feature_names)
    measurements = [x for x in df.columns if x != "Feature"]
    for m in measurements:
        plot_feature_selection(save_loc, m, condition, df, topn)
    print(df.nlargest(10, "Mutual Information"))


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
            print(df.nlargest(10, "Wasserstein Distance"))
            print(df["Wasserstein Distance"].quantile([0.25, 0.5, 0.75]))
            print()


def main(experiment_name, data_types):
    label = ["game"]
    parent_dir = "."
    if len(data_types) == 1:
        parent_dir = data_types[0]
    save_loc = get_data_path(parent_dir, f"model/{experiment_name}/features/fs")
    feature_df = read_and_clean_features(data_types, label, experiment_name)
    X, y, int_to_class, feature_names = df_to_xy(feature_df, label[0])
    X = scale(X, axis=0)

    run_feature_selection(save_loc, X, y, feature_names, "All", 5)
    run_pairwise_distributions(save_loc, X, y, int_to_class, feature_names, 5)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide a feature set and the data types to train the model with.")
