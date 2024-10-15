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
                                       mutual_info_classif, mutual_info_regression)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from common import get_colors
from spatial_statistics import (create_fsfr_features, create_pop_features, 
                                create_ripleysk_features, 
                                create_voroni_features, create_pc_features,
                                model_state_to_coords, read_model_state)
sys.path.insert(0, "DDIT")
from DDIT import DDIT

warnings.filterwarnings("ignore")
COLORS = get_colors()


'''
Aggregate HAL Runs
'''
def get_cell_counts(df_pop):
    df_pop = df_pop.loc[df_pop["time"] == df_pop["time"].max()]
    num_resistant = df_pop["resistant"].values[0]
    num_sensitive = df_pop["sensitive"].values[0]
    return num_sensitive, num_resistant


def save_data(exp_dir, dimension):
    df_entries = []
    exp_path = f"output/{exp_dir}"
    uid = 0
    extinct = 0
    unknwon_game = 0
    for game_dir in os.listdir(exp_path):
        game_path = f"{exp_path}/{game_dir}"
        if os.path.isfile(game_path):
            continue
        config = json.load(open(f"{game_path}/{game_dir}.json"))
        a = config["A"]
        b = config["B"]
        c = config["C"]
        d = config["D"]
        cells = config["numCells"]
        fr = config["proportionResistant"]
        for rep_dir in os.listdir(game_path):
            rep_path = f"{game_path}/{rep_dir}"
            if os.path.isfile(rep_path):
                continue
            pop_file = f"{rep_path}/{dimension}populations.csv"
            fs_file = f"{rep_path}/{dimension}fs.csv"
            fr_file = f"{rep_path}/{dimension}fr.csv"
            pc_file = f"{rep_path}/{dimension}pairCorrelations.csv"
            model_file = f"{rep_path}/{dimension}model250.csv" #TODO
            if not os.path.exists(pop_file) or os.path.getsize(pop_file) == 0:
                print(f"File not found in {rep_path}")
                continue
            sample_dict = {}
            sample_dict["rep"] = int(rep_dir)
            sample_dict["initial_fr"] = fr
            sample_dict["initial_cells"] = cells
            sample_dict["A"] = a
            sample_dict["B"] = b
            sample_dict["C"] = c
            sample_dict["D"] = d
            if a > c and b > d:
                game = "sensitive_wins"
            elif a < c and b > d:
                game = "coexistence"
            elif a > c and b < d:
                game = "bistability"
            elif a < c and b < d:
                game = "resistant_wins"
            else:
                game = "unknown"
            sample_dict["game"] = game
            uid += 1
            sample_dict["uid"] = uid
            df_pop = pd.read_csv(pop_file)
            num_sensitive, num_resistant = get_cell_counts(df_pop)
            if num_resistant < 100 or num_sensitive < 100:
                extinct += 1
                continue
            if game == "unknown":
                unknwon_game += 1
                continue
            df_fs = pd.read_csv(fs_file)
            df_fr = pd.read_csv(fr_file)
            df_pc = pd.read_csv(pc_file)
            model_state = read_model_state(model_file)
            feature_dict = create_all_features(df_fs, df_fr, df_pc, model_state, num_sensitive, num_resistant)
            sample_dict = sample_dict | feature_dict
            df_entries.append(sample_dict)
        if uid % 100 == 0:
            print(f"Processed {uid} samples...")
    print(f"Skipped {extinct} samples nearing extinction.")
    print(f"Skipped {unknwon_game} samples with unknown games.")
    print(f"Total samples: {len(df_entries)}")
    df = pd.DataFrame(data=df_entries)
    pd.to_pickle(df, f"output/{exp_dir}/{dimension}df.pkl")


'''
Feature Engineering
'''
def create_all_features(df_fs, df_fr, df_pc, model_state, num_sensitive, num_resistant):
    features = dict()
    features = features | create_pop_features(num_sensitive, num_resistant)
    features = features | create_fsfr_features(df_fs, df_fr, num_resistant, num_sensitive)
    features = features | create_pc_features(df_pc, num_sensitive, num_resistant)

    s_coords, r_coords = model_state_to_coords(model_state)
    features = features | create_ripleysk_features(s_coords, r_coords)
    features = features | create_voroni_features(s_coords, r_coords)
    
    return features


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
    fig.savefig(f"output/{exp_dir}/correlations.png", bbox_inches="tight")
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
        clf = MLPClassifier(hidden_layer_sizes=(400,200,100,50)).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = round(sum([y_pred[i] == y_test[i] for i in range(len(y_test))])/len(y_test), 2)
        avg_acc += acc
        conf_mat = confusion_matrix(y_test, y_pred, normalize="true")
        disp_labels = [int_to_category[x] for x in clf.classes_]
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=disp_labels)
        disp.plot()
        plt.savefig(f"output/{exp_dir}/confusion_matrix{k}.png", bbox_inches="tight")
        print(f"\tAccuracy {k}: {acc}")
    print("\tAverage Accuracy:", avg_acc/5)


'''
Run Script
'''
def main(exp_dir, dimension):
    print("Reading in data...")
    try:
        df = pd.read_pickle(f"output/{exp_dir}/{dimension}df.pkl")
    except:
        print("Please save the dataframe.")
        exit()
    nonfeature_cols = ["uid", "rep", "initial_fr", "initial_cells",
                       "A", "B", "C", "D", "game"]

    classify_game = True
    labels = ["game"] if classify_game else ["A", "B", "C", "D"]
    #feature_names = [x for x in df.columns if x not in nonfeature_cols]
    feature_names = ["fs_mean", "fs_std", "fs_slope", "fs_skew", "pc_SR_mean"]
    features = df[feature_names+labels]

    print("\nAnalyzing and exploring data...")
    feature_correlation(exp_dir, features, labels)
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