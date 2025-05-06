"""Exploratory feature analysis

Plots that will be generated:
    Correlated clusters
    Feature distribution ridgeplots
    Correlation matrix
    Pairplot

Expected usage:
python3 -m spatial_egt.classification.feature_exploration data_type label_name feature_names

Where:
data_type: the name of the directory in data/ containing spatial_statistics/features.csv
label_name: the class label name, which is also a column in data/{data_type}/labels.csv
feature_names: a list of the feature names to explore, or "all" or "noncorr" which are pre-defined
"""

import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import networkx as nx
import numpy as np
from scipy import stats
from scipy.sparse import csgraph, csr_matrix
import seaborn as sns

from spatial_egt.classification.common import get_feature_data
from spatial_egt.common import game_colors, theme_colors


def feature_pairplot(save_loc, df, label_hue):
    sns.pairplot(df, hue=label_hue)
    plt.savefig(f"{save_loc}/feature_pairplot_{label_hue}.png", bbox_inches="tight")
    plt.close()


def features_ridgeplots(save_loc, df, feature_names, label_name):
    '''
    Based on https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    '''
    class_names = sorted(df[label_name].unique())
    colors = sns.color_palette("hls", len(df[label_name].unique()))
    if label_name == "game":
        games = {k:v for k,v in game_colors.items() if k in class_names}
        colors = list(games.values())
        class_names = games.keys()
    num_features = len(feature_names)
    num_classes = len(class_names)
    gs = grid_spec.GridSpec(num_classes, num_features)
    fig = plt.figure(figsize=(6*num_features, 8))
    axes = []
    for f,feature_name in enumerate(feature_names):
        min_val = df[feature_name].min()
        x = np.linspace(min_val, df[feature_name].max(), 100)
        for c,class_name in enumerate(class_names):
            feature_class_data = df.loc[df[label_name] == class_name][feature_name]
            kde = stats.gaussian_kde(feature_class_data)
            axes.append(fig.add_subplot(gs[c:c+1, f:f+1]))
            axes[-1].plot(x, kde(x), color="white")
            axes[-1].fill_between(x, kde(x), alpha=0.75, color=colors[c])
            rect = axes[-1].patch
            rect.set_alpha(0)
            axes[-1].set_yticklabels([])
            if c == num_classes-1:
                axes[-1].set_xlabel(feature_name.replace("_", " "))
            else:
                axes[-1].set(xticklabels=[], xticks=[])
            if f == 0:
                sd = np.std(df[feature_name])
                axes[-1].text(min_val - sd/3, 0, class_name, ha="right")
            else:
                axes[-1].set(yticklabels=[])
            axes[-1].set(yticks=[])
            spines = ["top", "right", "left", "bottom"]
            for s in spines:
                axes[-1].spines[s].set_visible(False)
    gs.update(hspace=-0.7)
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/feature_ridgeplot.png", bbox_inches="tight")
    plt.close()


def feature_correlation(save_loc, df, feature_names):
    num_features = len(feature_names)
    fig, ax = plt.subplots(figsize=(12, 12))
    correlation_matrix = df[feature_names].corr(method="spearman")
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
    fig.savefig(f"{save_loc}/corr_matrix.png", bbox_inches="tight")
    plt.close()


def visualize_correlated(save_loc, df, feature_names, print_latex=False):
    corr_matrix = df[feature_names].corr(method="spearman")
    high_corr = ((corr_matrix >= 0.9) | (corr_matrix <= -0.9)) & (corr_matrix != 1.0)
    adj_matrix = csr_matrix(high_corr)
    _, labels = csgraph.connected_components(csgraph=adj_matrix, directed=False)
    clusters = [[] for _ in range(len(set(labels)))]
    for i, label in enumerate(labels):
        clusters[label].append(i)

    feature_names = [x.replace("_", " ") for x in feature_names]
    print("Correlated Clusters")
    for c in clusters:
        print("\t", [feature_names[i] for i in c])

    if print_latex:
        print("\\textbf{Chosen Feature} & \\textbf{Features in Correlated Cluster} \\\\")
        for c in clusters:
            print("\\hline")
            print(f"{feature_names[c[0]]} & {', '.join([feature_names[i] for i in c[1:]])} \\\\")
        print("\\hline")

    for cluster in [x for x in clusters if len(x) > 1]:
        cluster_name = feature_names[cluster[0]]
        fig, ax = plt.subplots(figsize=(10,10))
        graph = nx.Graph(adj_matrix)
        graph = graph.subgraph(cluster)
        labels = {i:feature_names[i].replace(" ", "\n") for i in cluster}
        nx.draw(graph, pos=nx.kamada_kawai_layout(graph), labels=labels,
                node_size=3000, font_size=10, node_color=theme_colors[1], ax=ax)
        fig.patch.set_alpha(0.0)
        fig.tight_layout()
        fig.savefig(f"{save_loc}/corr_graph_{cluster_name}.png", bbox_inches="tight")
        plt.close()


def main(data_type, label_name, feature_names):
    save_loc, df, feature_names = get_feature_data(data_type, label_name, feature_names, "statistics")
    feature_df = df[feature_names+[label_name]]

    visualize_correlated(save_loc, df, feature_names, False)
    feature_correlation(save_loc, feature_df, feature_names)
    if len(feature_names) <= 30:
        features_ridgeplots(save_loc, feature_df, feature_names, label_name)
    if len(feature_names) <= 20:
        feature_pairplot(save_loc, feature_df, label_name)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        print("Please provide the data type, label name, and feature set/names.")
