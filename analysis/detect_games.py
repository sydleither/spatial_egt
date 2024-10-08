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
                                       mutual_info_classif, mutual_info_regression,
                                       SequentialFeatureSelector)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from common import get_colors, process_fs
sys.path.insert(0, "DDIT")
from DDIT import DDIT

warnings.filterwarnings("ignore")
COLORS = get_colors()


'''
Aggregate HAL Runs
'''
def save_data(exp_dir, dimension):
    df = pd.DataFrame()
    exp_path = f"output/{exp_dir}"
    uid = 0
    # for grid_params in os.listdir(exp_path):
    #     grid_path = f"{exp_path}/{grid_params}"
    #     if os.path.isfile(grid_path):
    #         continue
    for game_dir in os.listdir(exp_path):
        game_path = f"{exp_path}/{game_dir}"
        if os.path.isfile(game_path):
            continue
        config = json.load(open(f"{game_path}/{game_dir}.json"))
        a = config["A"]
        b = config["B"]
        c = config["C"]
        d = config["D"]
        grid_size = config["x"]*config["y"]
        cells = config["numCells"]
        fr = config["proportionResistant"]
        for rep_dir in os.listdir(game_path):
            rep_path = f"{game_path}/{rep_dir}"
            if os.path.isfile(rep_path):
                continue
            pop_file = f"{rep_path}/{dimension}populations.csv"
            fs_file = f"{rep_path}/{dimension}fs.csv"
            if not os.path.exists(pop_file) or os.path.getsize(pop_file) == 0 or\
                not os.path.exists(fs_file) or os.path.getsize(fs_file) == 0:
                print(f"File not found in {rep_path}")
                continue
            df_pop_i = pd.read_csv(pop_file)
            df_pop_i["dimension"] = dimension
            df_pop_i["rep"] = int(rep_dir)
            df_pop_i["grid_size"] = grid_size
            df_pop_i["initial_fr"] = fr
            df_pop_i["initial_cells"] = cells
            df_pop_i["A"] = a
            df_pop_i["B"] = b
            df_pop_i["C"] = c
            df_pop_i["D"] = d
            df_pop_i["uid"] = uid
            df_fs_i = pd.read_csv(fs_file)
            df_fs_i = process_fs(df_fs_i, ["model", "time", "radius"])
            df_i = df_fs_i.merge(df_pop_i, on=["model", "time"])
            df = pd.concat([df, df_i])
            uid += 1
    df["game"] = "unknown"
    df.loc[(df["A"] > df["C"]) & (df["B"] > df["D"]), "game"] = "sensitive_wins"
    df.loc[(df["A"] < df["C"]) & (df["B"] > df["D"]), "game"] = "coexistence"
    df.loc[(df["A"] > df["C"]) & (df["B"] < df["D"]), "game"] = "bistability"
    df.loc[(df["A"] < df["C"]) & (df["B"] < df["D"]), "game"] = "resistant_wins"
    pd.to_pickle(df, f"output/{exp_dir}/{dimension}df.pkl")


'''
Feature Engineering
'''
def create_fs_features(df):
    #slope of average fs over neighborhood radii
    df_r1 = df.loc[df["radius"] == 1]
    df_r1["avg_fs_r1"] = df_r1["average_fs"]
    df_r5 = df.loc[df["radius"] == 5]
    df_r5["avg_fs_r5"] = df_r5["average_fs"]
    df_slope = df_r1[["uid", "avg_fs_r1"]].merge(df_r5[["uid", "avg_fs_r5"]], on=["uid"])
    df_slope = df_slope.drop_duplicates()
    df_slope["avg_fs_slope"] = df_slope["avg_fs_r5"] - df_slope["avg_fs_r1"]
    df_slope = df_slope.drop(columns=["avg_fs_r1", "avg_fs_r5"])
    df = df.merge(df_slope, on=["uid"])

    #proportion of boundary cells
    df_r1 = df.loc[df["radius"] == 1]
    df_r1["s_in_neighborhood"] = df_r1["total_boundary"] / df_r1["resistant"]
    df_r1 = df_r1[["uid", "s_in_neighborhood"]].drop_duplicates()
    df = df.merge(df_r1, on=["uid"])

    #average fs
    df_r3 = df.loc[df["radius"] == 3]
    df = df.drop(columns=["average_fs"])
    df_r3 = df_r3[["uid", "average_fs"]].drop_duplicates()
    df = df.merge(df_r3, on=["uid"])

    #fs distribution summary statistics
    df_r3 = df.loc[df["radius"] == 3]
    fs_expand = pd.DataFrame({
        "uid": np.repeat(df_r3["uid"], df_r3["total"]),
        "fs": np.repeat(df_r3["fs"], df_r3["total"])
    })
    fs_stats = fs_expand.groupby("uid")["fs"].agg(["skew", "std"])
    fs_stats = fs_stats.rename(columns={"skew":"skew_fs",
                                        "std":"std_fs"})
    df = df.merge(fs_stats, on=["uid"])

    #collaspe (one row for one run)
    df = df.drop(columns=["radius", "fs", "total", "reproduced", "weighted_fs", 
                          "total_boundary", "weighted_fs_sum", "normalized_total"])
    df = df.drop_duplicates()
    
    return df


def create_pop_features(df):
    #proportion of cells that are resistant
    df["proportion_resistant"] = df["resistant"] / (df["resistant"] + df["sensitive"])

    #amount of grid filled
    df["density"] = (df["resistant"] + df["sensitive"]) / df["grid_size"]

    return df


def clean_data(df):
    #remove unclassifiable games (on the border of the game space)
    org_size = len(df)
    df = df.loc[df["game"] != "unknown"]
    print(f"\tRemoved {org_size-len(df)} unknown games.")

    #remove grids with basically extinct cell lines
    org_size = len(df)
    df = df.loc[(df["sensitive"] >= 100) & (df["resistant"] >= 100)]
    print(f"\tRemoved {org_size-len(df)} approaching extinction cell lines.")

    return df


'''
Data Exploration / Visualization
'''
def feature_pairplot(exp_dir, df, label_hue):
    sns.pairplot(df, hue=label_hue)
    plt.savefig(f"output/{exp_dir}/feature_pairplot_{label_hue}.png", bbox_inches="tight")
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
    fig.savefig(f"output/{exp_dir}/feature_labels{num_labels}.png", bbox_inches="tight")
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
    for l,label_name in enumerate(label_names):
        label_entropy = ddit.H(label_name)
        print(f"\t{label_name} entropy: {label_entropy}")
        print(f"\t\tideal: log({len(df[label_name].unique())}) = {np.log2(len(df[label_name].unique()))}")
        for feature_set in feature_powerset:
            ent = ddit.recursively_solve_formula(label_name+":"+"&".join(feature_set)) / label_entropy
            entropies[l].append(ent)

    #visualize
    num_feature_sets = len(entropies[0])
    fig, ax = plt.subplots(figsize=(15,5))
    ax.imshow(np.array(entropies), cmap="Greens")
    ax.set_xticks(np.arange(num_feature_sets), labels=["".join(x) for x in feature_powerset])
    ax.set_yticks(np.arange(num_labels), labels=label_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for l in range(num_labels):
        for j in range(num_feature_sets):
            ax.text(j, l, round(entropies[l][j], 2), ha="center", va="center", color="hotpink")
    ax.set_title("Fragmentation Matrix")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    fig.savefig(f"output/{exp_dir}/fragmentation{num_labels}.png", bbox_inches="tight")
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

        print(f"\tSequential Feature Selection {label_name}")
        clf = MLPClassifier(hidden_layer_sizes=(100,75,50))
        sfs = SequentialFeatureSelector(clf, tol=0.05, cv=5).fit(X, y)
        print(f"\t\t{sfs.get_feature_names_out()}")


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
    for _, (train_i, test_i) in enumerate(cross_validation.split(X, y)):
        X_train = [X[i] for i in train_i]
        X_test = [X[i] for i in test_i]
        y_train = [y[i] for i in train_i]
        y_test = [y[i] for i in test_i]
        clf = MLPClassifier(hidden_layer_sizes=(400,200,100,50)).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = round(sum([y_pred[i] == y_test[i] for i in range(len(y_test))])/len(y_test), 2)
        avg_acc += acc
        print("Accuracy:", acc)
        print(confusion_matrix(y_test, y_pred, normalize="true"))
    print("Average Accuracy:", avg_acc/5)
    print(int_to_category)


def main(exp_dir, dimension):
    print("Reading in data...")
    try:
        df = pd.read_pickle(f"output/{exp_dir}/{dimension}df.pkl")
    except:
        print("Please save the dataframe.")
        exit()

    print("\nCreating features and cleaning data...")
    df = create_pop_features(create_fs_features(df))
    df = clean_data(df)

    classify_game = True
    labels = ["game"] if classify_game else ["A", "B", "C", "D"]
    #feature_names = ['s_in_neighborhood', 'skew_fs', 'std_fs']
    feature_names = ["s_in_neighborhood", "average_fs", "proportion_resistant", "skew_fs", "std_fs", "avg_fs_slope", "density"]
    features = df[feature_names+labels]

    print("\nAnalyzing and exploring data...")
    features_by_labels_plot(exp_dir, features, labels)
    fragmentation_matrix_plot(exp_dir, features, labels, "equal")
    feature_selection(features, labels)
    if classify_game:
        feature_pairplot(exp_dir, features, labels[0])
    else:
        feature_pairplot(exp_dir, features, None)

    print("\nRunning machine learning...")
    if classify_game:
        machine_learning(exp_dir, features, labels[0])


if __name__ == "__main__":
    if len(sys.argv) == 4:
        if sys.argv[3] == "save":
            save_data(sys.argv[1], sys.argv[2])
        else:
            print("Please provide am experiment directory, dimension, and \"save\"")
            print("if the dataframe has not yet been saved.")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the experiment directory and dimension, if the dataframe has been saved.")