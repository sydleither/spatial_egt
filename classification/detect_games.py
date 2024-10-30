from itertools import chain, combinations
import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import (f_classif, f_regression, 
                                       mutual_info_classif, 
                                       mutual_info_regression)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from common import get_data_path
from data_analysis.DDIT.DDIT import DDIT

warnings.filterwarnings("ignore")
COLORS = ["#4C956C", "#EF7C8E", "#f97306", "#029386"]


'''
Data Exploration / Visualization
'''
def feature_pairplot(exp_dir, df, label_hue):
    sns.pairplot(df, hue=label_hue)
    plt.savefig(f"{exp_dir}/feature_pairplot_{label_hue}.png", bbox_inches="tight")
    plt.close()


def features_by_labels_plot(exp_dir, df, label_names):
    feature_names = list(df.columns)
    [feature_names.remove(ln) for ln in label_names]
    num_features = len(feature_names)
    num_labels = len(label_names)
    fig, ax = plt.subplots(num_labels, num_features, figsize=(8*num_features,8*num_labels))
    for l,label_name in enumerate(label_names):
        label_dtype = df[label_name].dtypes
        for f,feature_name in enumerate(feature_names):
            axis = ax[f] if num_labels == 1 else ax[l][f]
            if label_dtype == float:
                sns.scatterplot(data=df, x=feature_name, y=label_name, 
                                color=COLORS[0], ax=axis)
            else:
                sns.boxplot(data=df, x=feature_name, y=label_name, hue=label_name, 
                            legend=False, notch=True, palette=COLORS, ax=axis)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"{exp_dir}/feature_labels{num_labels}.png", bbox_inches="tight")
    plt.close()


def feature_correlation(exp_dir, df, label_names):
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
    fig.savefig(f"{exp_dir}/correlations.png", bbox_inches="tight")
    plt.close()


def fragmentation_matrix_plot(exp_dir, df, label_names, binning_method):
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
    fig.savefig(f"{exp_dir}/fragmentation{num_labels}.png", bbox_inches="tight")
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
            int_to_category = {i:lc for i,lc in enumerate(label_categories)}
            y = [category_to_int[x] for x in df[label_name].values]
            mutual_info = mutual_info_classif(X, y)
            f_statistic, p_values = f_classif(X, y)

        print(f"\tMutual Information {label_name}")
        for i in range(len(feature_names)):
            print(f"\t\t{feature_names[i]} info:{mutual_info[i]}")

        print(f"\tANOVA F-Statistics {label_name}")
        for i in range(len(feature_names)):
            print(f"\t\t{feature_names[i]} F:{round(f_statistic[i])} p-value:{p_values[i]}")


'''
Classification
'''
def machine_learning(exp_dir, df, label_name):
    feature_names = list(df.columns)
    feature_names.remove(label_name)
    label_categories = df[label_name].unique()
    category_to_int = {lc:i for i,lc in enumerate(label_categories)}
    int_to_category = {i:lc for i,lc in enumerate(label_categories)}
    X = list(df[feature_names].values)
    y = [category_to_int[x] for x in df[label_name].values]

    avg_acc = 0
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True)
    for k, (train_i, test_i) in enumerate(cross_validation.split(X, y)):
        X_train = [X[i] for i in train_i]
        X_test = [X[i] for i in test_i]
        y_train = [y[i] for i in train_i]
        y_test = [y[i] for i in test_i]
        clf = MLPClassifier(hidden_layer_sizes=(500,250,100,50)).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = round(sum([y_pred[i] == y_test[i] for i in range(len(y_test))])/len(y_test), 2)
        avg_acc += acc
        conf_mat = confusion_matrix(y_test, y_pred, normalize="true")
        disp_labels = [int_to_category[x] for x in clf.classes_]

        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=disp_labels)
        disp.plot(ax=ax)
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(15)
        ax.set_title(f"Accuracy: {acc}")
        fig.tight_layout()
        fig.savefig(f"{exp_dir}/confusion_matrix{k}.png", bbox_inches="tight", transparent=True)
        print(f"\tAccuracy {k}: {acc}")
    print("\tAverage Accuracy:", avg_acc/5)


'''
Run Script
'''
def main():
    exp_dir = get_data_path("in_silico", "images")
    print("Reading in data...")
    try:
        features_data_path = get_data_path("in_silico", "features")
        df = pd.read_csv(f"{features_data_path}/features.csv")
    except:
        print("Please save the feature dataframe.")
        exit()

    df = df[df["game"] != "unknown"]
    df = df[df["proportion_s"] != 1]
    df = df[df["proportion_s"] != 0]
    df = df.dropna()

    labels = ["game"]
    feature_names = ["fs_mean", "fs_std", "fs_skew", "fr_mean", "fr_std", "fr_skew"]
    features = df[feature_names+labels]

    # print("\nAnalyzing and exploring data...")
    # feature_correlation(exp_dir, features, labels)
    # features_by_labels_plot(exp_dir, features, labels)
    # fragmentation_matrix_plot(exp_dir, features, labels, "equal")
    # feature_selection(features, labels)
    # feature_pairplot(exp_dir, features, labels[0])

    print("\nRunning machine learning...")
    machine_learning(exp_dir, features, labels[0])


if __name__ == "__main__":
    main()