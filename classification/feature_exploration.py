from collections import Counter
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import numpy as np
from scipy import stats
import seaborn as sns

from classification.common import read_and_clean_features, remove_correlated
from common import game_colors, get_data_path


def feature_pairplot(save_loc, df, label_hue):
    sns.pairplot(df, hue=label_hue)
    plt.savefig(f"{save_loc}/feature_pairplot_{label_hue}.png", bbox_inches="tight")
    plt.close()


def features_ridgeplots(save_loc, df, label_names, colors, label_orders):
    '''
    Based on https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    '''
    feature_names = list(df.columns)
    [feature_names.remove(ln) for ln in label_names]
    num_features = len(feature_names)

    for label_name in label_names:
        label_dtype = df[label_name].dtypes
        if label_dtype == float:
            continue
        class_labels = label_orders[label_name]
        num_classes = len(class_labels)
        gs = (grid_spec.GridSpec(num_classes, num_features))
        fig = plt.figure(figsize=(6*num_features, 8))
        axes = []
        for f,feature_name in enumerate(feature_names):
            x = np.linspace(df[feature_name].min(), df[feature_name].max(), 100)
            for c,class_name in enumerate(class_labels):
                feature_class_data = df.loc[df[label_name] == class_name][feature_name]
                kde = stats.gaussian_kde(feature_class_data)
                axes.append(fig.add_subplot(gs[c:c+1, f:f+1]))
                axes[-1].plot(x, kde(x), color="white")
                axes[-1].fill_between(x, kde(x), alpha=0.75, color=colors[class_name])
                rect = axes[-1].patch
                rect.set_alpha(0)
                axes[-1].set_yticklabels([])
                if c == num_classes-1:
                    axes[-1].set_xlabel(feature_name)
                else:
                    axes[-1].set(xticklabels=[], xticks=[])
                if f == 0:
                    axes[-1].text(-0.02, 0, class_name, ha="right")
                else:
                    axes[-1].set(yticklabels=[])
                axes[-1].set(yticks=[])
                spines = ["top", "right", "left", "bottom"]
                for s in spines:
                    axes[-1].spines[s].set_visible(False)
        gs.update(hspace=-0.7)
        fig.tight_layout()
        fig.figure.patch.set_alpha(0.0)
        fig.savefig(f"{save_loc}/feature_ridgeplot_{label_name}.png", bbox_inches="tight")
        plt.close()


def features_by_labels_plot(save_loc, df, label_names, colors, color_order):
    feature_names = list(df.columns)
    [feature_names.remove(ln) for ln in label_names]
    num_features = len(feature_names)
    num_labels = len(label_names)

    fig, ax = plt.subplots(num_labels, num_features, figsize=(7*num_features, 7*num_labels))
    if num_features == 1:
        ax = [ax]
    for l,label_name in enumerate(label_names):
        label_dtype = df[label_name].dtypes
        for f,feature_name in enumerate(feature_names):
            axis = ax[f] if num_labels == 1 else ax[l][f]
            if label_dtype == float:
                sns.scatterplot(data=df, x=feature_name, y=label_name, 
                                color=colors[0], ax=axis)
            else:
                sns.violinplot(data=df, x=feature_name, y=label_name,
                               cut=0, legend=False, ax=axis,
                               order=color_order, palette=colors)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/feature_labels{num_labels}.png", bbox_inches="tight")
    plt.close()


def feature_correlation(save_loc, df, label_names):
    feature_names = list(df.columns)
    [feature_names.remove(ln) for ln in label_names]
    num_features = len(feature_names)

    fig, ax = plt.subplots(figsize=(12, 12))
    correlation_matrix = df[feature_names].corr()
    ax.imshow(correlation_matrix, cmap="PiYG")
    ax.set_xticks(np.arange(num_features), labels=feature_names)
    ax.set_yticks(np.arange(num_features), labels=feature_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for l,name1 in enumerate(feature_names):
        for j,name2 in enumerate(feature_names):
            ax.text(j, l, round(correlation_matrix[name1][name2], 2), ha="center", va="center")
    ax.set_title("Correlation Matrix")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/correlations.png", bbox_inches="tight")
    plt.close()


def class_balance(df, label_names):
    for label_name in label_names:
        print(label_name)
        counts = dict(Counter(df[label_name]))
        total = sum(counts.values())
        proportions = {k:round(v/total, 3) for k,v in counts.items()}
        print(f"\tTotal: {total}")
        print(f"\tCounts: {counts}")
        print(f"\tProportions: {proportions}")


def print_correlated_clusters(df, label_names):
    feature_names = sorted(list(df.columns))
    [feature_names.remove(ln) for ln in label_names]

    corr_matrix = df[feature_names].corr()
    non_correlated = remove_correlated(df, label_names)
    for feature in non_correlated:
        row = corr_matrix[feature]
        correlated = list(row[((row >= 0.9) | (row <= -0.9)) & (row != 1)].index)
        print(f"{feature}: {correlated}")


def main(experiment_name, data_type):
    label = ["game"]
    feature_df = read_and_clean_features([data_type], label, experiment_name)
    save_loc = get_data_path(data_type, f"model/{experiment_name}/features")

    print_correlated_clusters(feature_df, label)
    features_ridgeplots(save_loc, feature_df, label, game_colors,
                        label_orders={"game":game_colors.keys()})
    class_balance(feature_df, label)
    feature_correlation(save_loc, feature_df, label)
    features_by_labels_plot(save_loc, feature_df, label, 
                            game_colors.values(), game_colors.keys())
    feature_pairplot(save_loc, feature_df, label[0])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide a feature set and the data type.")
