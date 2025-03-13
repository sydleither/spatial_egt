import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (f_classif,
                                       mutual_info_classif)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

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
    df = df.sort_values(measurement, ascending=False)
    df["Feature"] = df["Feature"].str.replace("_", " ")
    df["Statistic"] = df["Feature"].map(color_by_statistic(df["Feature"].unique()))
    fig, ax = plt.subplots(figsize=(6, 12))
    sns.barplot(data=df, x=measurement, y="Feature", ax=ax,
                hue="Statistic", palette=sns.color_palette("Set2"),
                hue_order=sorted(df["Statistic"].unique()))
    ax.set(title=f"Feature {measurement}\n{condition}")
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/{measurement}_{condition}.png", bbox_inches="tight")
    plt.close()


def rf_importance(X, y, n_repeats=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    rf = RandomForestClassifier().fit(X_train, y_train)
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


def run_feature_selection(save_loc, X, y, feature_names, condition):
    df = feature_selection(X, y, feature_names)
    measurements = [x for x in df.columns if x != "Feature"]
    for m in measurements:
        plot_feature_selection(save_loc, m, condition, df)


def main(experiment_name, *data_types):
    label = ["game"]
    parent_dir = "."
    if len(data_types[0]) == 1:
        parent_dir = data_types[0][0]
    save_loc = get_data_path(parent_dir, f"model/{experiment_name}/features/fs")
    feature_df = read_and_clean_features(data_types[0], label, experiment_name)
    X, y, int_to_class, feature_names = df_to_xy(feature_df, label[0])

    run_feature_selection(save_loc, X, y, feature_names, "All")

    for i,game in int_to_class.items():
        y_game = [1 if label == i else 0 for label in y]
        run_feature_selection(save_loc, X, y_game, feature_names, game)

    for i in range(len(int_to_class)):
        for j in range(i+1, len(int_to_class)):
            indices = [k for k in range(len(y)) if y[k] == i or y[k] == j]
            X_pair = [X[k] for k in indices]
            y_pair = [y[k] for k in indices]
            pair_name = f"{int_to_class[i]} - {int_to_class[j]}"
            run_feature_selection(save_loc, X_pair, y_pair, feature_names, pair_name)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide a feature set and the data types to train the model with.")
