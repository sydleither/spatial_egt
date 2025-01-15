from collections import Counter
from itertools import chain, combinations
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import (f_classif, f_regression, 
                                       mutual_info_classif, 
                                       mutual_info_regression)

from common.common import game_colors, get_data_path
from common.classification import clean_feature_data, features
from data_analysis.DDIT.DDIT import DDIT

warnings.filterwarnings("ignore")


'''
Data Exploration / Visualization
'''
def feature_pairplot(save_loc, df, label_hue):
    sns.pairplot(df, hue=label_hue)
    plt.savefig(f"{save_loc}/feature_pairplot_{label_hue}.png", bbox_inches="tight")
    plt.close()


def features_by_labels_plot(save_loc, df, label_names, colors, color_order):
    feature_names = list(df.columns)
    [feature_names.remove(ln) for ln in label_names]
    num_features = len(feature_names)
    num_labels = len(label_names)
    fig, ax = plt.subplots(num_labels, num_features, figsize=(7*num_features, 7*num_labels))
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

    fig, ax = plt.subplots()
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


def fragmentation_matrix_plot(save_loc, df, label_names, binning_method):
    #initializations
    ddit = DDIT()
    feature_names = list(df.columns)
    [feature_names.remove(ln) for ln in label_names]
    num_labels = len(label_names)
    num_features = len(feature_names)
    feature_name_map = {name:str(i) for i,name in enumerate(feature_names)}

    #bin and register features
    for feature_name in feature_names:
        feature_name_index = feature_name_map[feature_name]
        column_data = df[feature_name].values
        if binning_method == "round":
            column_data = [round(x,2) for x in column_data]
        elif binning_method == "equal":
            _, bin_edges = np.histogram(column_data, bins=10)
            column_data = np.digitize(column_data, bin_edges)
        else:
            print("Invalid binning method  provided to create_fragmentation_matrix().")
            return
        ddit.register_column_tuple(feature_name_index, tuple(column_data))
    for ln in label_names:
        ddit.register_column_tuple(ln, tuple(df[ln].values))
    
    #calculate entropies
    feature_powerset = chain.from_iterable(combinations(feature_name_map.values(), r) for r in range(num_features+1))
    feature_powerset = list(feature_powerset)[1:]
    entropies = [[] for _ in range(num_labels)]
    valid_feature_sets = []
    for l,label_name in enumerate(label_names):
        label_entropy = ddit.H(label_name)
        print(f"\t{label_name} entropy: {label_entropy}")
        print(f"\t\tideal: log({len(df[label_name].unique())}) = {np.log2(len(df[label_name].unique()))}")
        for feature_set in feature_powerset:
            if len(feature_set) > 2 and len(feature_set) < num_features and num_features > 6:
                continue
            ent = ddit.recursively_solve_formula(label_name+":"+"&".join(feature_set)) / label_entropy
            entropies[l].append(ent)
            if l == 0:
                valid_feature_sets.append("".join(feature_set))

    #visualize
    num_feature_sets = len(entropies[0])
    fig, ax = plt.subplots(figsize=(30,5))
    ax.imshow(np.array(entropies), cmap="Greens")
    ax.set_xticks(np.arange(num_feature_sets), labels=valid_feature_sets)
    ax.set_yticks(np.arange(num_labels), labels=label_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for l in range(num_labels):
        for j in range(num_feature_sets):
            ax.text(j, l, round(entropies[l][j], 2), ha="center", va="center", color="hotpink")
    ax.set_title(f"Fragmentation Matrix\n{feature_name_map}")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/fragmentation{num_labels}.png", bbox_inches="tight")
    plt.close()


def feature_selection(df, label_names):
    feature_names = list(df.columns)
    [feature_names.remove(ln) for ln in label_names]

    for label_name in label_names:
        label_dtype = df[label_name].dtypes
        X = df[feature_names]
        
        if label_dtype == float:
            y = np.array(df[label_name].values)
            mutual_info = mutual_info_regression(X, y)
            f_statistic, p_values = f_regression(X, y)
        else:
            label_categories = df[label_name].unique()
            category_to_int = {lc:i for i,lc in enumerate(label_categories)}
            y = [category_to_int[x] for x in df[label_name].values]
            mutual_info = mutual_info_classif(X, y)
            f_statistic, p_values = f_classif(X, y)

        print(f"\tMutual Information {label_name}")
        for i in range(len(feature_names)):
            print(f"\t\t{feature_names[i]} info:{mutual_info[i]}")

        print(f"\tANOVA F-Statistics {label_name}")
        for i in range(len(feature_names)):
            print(f"\t\t{feature_names[i]} F:{round(f_statistic[i])} p-value:{p_values[i]}")


def class_balance(df, label_names):
    for label_name in label_names:
        print(label_name)
        counts = dict(Counter(df[label_name]))
        total = sum(counts.values())
        proportions = {k:round(v/total, 3) for k,v in counts.items()}
        print(f"\tTotal: {total}")
        print(f"\tCounts: {counts}")
        print(f"\tProportions: {proportions}")


def main(data_type):
    features_data_path = get_data_path(data_type, "features")
    images_data_path = get_data_path(data_type, "images")
    df = pd.read_csv(f"{features_data_path}/all.csv")
    df = clean_feature_data(df)

    label = ["game"]
    if len(features) == 0:
        feature_df = df
    else:
        feature_df = df[features+label]
    
    class_balance(df, label)
    feature_correlation(images_data_path, feature_df, label)
    features_by_labels_plot(images_data_path, feature_df, label, 
                            game_colors.values(), game_colors.keys())
    fragmentation_matrix_plot(images_data_path, feature_df, label, "equal")
    feature_selection(feature_df, label)
    feature_pairplot(images_data_path, feature_df, label[0])


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the data type to analyze.")