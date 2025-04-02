from collections import Counter
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import networkx as nx
import numpy as np
from scipy import stats
from scipy.sparse import csgraph, csr_matrix
import seaborn as sns

from classification.common import get_feature_data
from common import game_colors, theme_colors


def feature_pairplot(save_loc, df, label_hue):
    sns.pairplot(df, hue=label_hue)
    plt.savefig(f"{save_loc}/feature_pairplot_{label_hue}.png", bbox_inches="tight")
    plt.close()


def features_ridgeplots(save_loc, df, feature_names, label_name, colors):
    '''
    Based on https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    '''
    num_features = len(feature_names)
    class_names = df[label_name].unique()
    num_classes = len(class_names)
    gs = (grid_spec.GridSpec(num_classes, num_features))
    fig = plt.figure(figsize=(6*num_features, 8))
    axes = []
    for f,feature_name in enumerate(feature_names):
        x = np.linspace(df[feature_name].min(), df[feature_name].max(), 100)
        for c,class_name in enumerate(class_names):
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


def class_balance(df, label_name):
    print(label_name)
    counts = dict(Counter(df[label_name]))
    total = sum(counts.values())
    proportions = {k:round(v/total, 3) for k,v in counts.items()}
    print(f"\tTotal: {total}")
    print(f"\tCounts: {counts}")
    print(f"\tProportions: {proportions}")


def visualize_correlated(save_loc, df, feature_names):
    corr_matrix = df[feature_names].corr(method="spearman")
    high_corr = ((corr_matrix >= 0.9) | (corr_matrix <= -0.9)) & (corr_matrix != 1.0)
    adj_matrix = csr_matrix(high_corr)
    _, labels = csgraph.connected_components(csgraph=adj_matrix, directed=False)
    clusters = [[] for _ in range(len(set(labels)))]
    for i, label in enumerate(labels):
        clusters[label].append(i)

    print("Correlated Clusters")
    for c in clusters:
        print("\t", [feature_names[i] for i in c])

    prop_s = max(clusters, key=len)
    fig, ax = plt.subplots(figsize=(10,10))
    graph = nx.Graph(adj_matrix)
    graph = graph.subgraph(prop_s)
    labels = {i:feature_names[i].replace("_", "\n") for i in prop_s}
    nx.draw(graph, pos=nx.kamada_kawai_layout(graph), labels=labels,
            node_size=2000, font_size=8, node_color=theme_colors[0], ax=ax)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/corr_ps_graph.png", bbox_inches="tight")
    plt.close()


def main(data_type, feature_names):
    save_loc, df, feature_names, label_name = get_feature_data(data_type, feature_names, "features")
    feature_df = df[feature_names+[label_name]]
    colors = {k:v for k,v in game_colors.items() if k in feature_df["game"].unique()}

    # df["cell_type"] = df["source"].apply(lambda x: "_".join(x.split("_")[1:-1]).lower())
    # c = {s:"purple" for s in df["cell_type"].unique()}
    # s = {"cell_type":list(df["cell_type"].unique())}
    # df = df.drop(["source", "sample", "game"], axis=1)
    # features_ridgeplots(save_loc, df, ["cell_type"], c, s)

    visualize_correlated(save_loc, df, feature_names)
    features_ridgeplots(save_loc, feature_df, feature_names, label_name, colors)
    class_balance(feature_df, label_name)
    feature_correlation(save_loc, feature_df, feature_names)
    feature_pairplot(save_loc, feature_df, label_name)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide the data type and the feature set/names.")
