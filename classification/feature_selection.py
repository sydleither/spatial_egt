'''
Based on: https://scikit-learn.org/stable/modules/feature_selection.html
'''
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (f_classif,
                                       mutual_info_classif,
                                       RFECV,
                                       SequentialFeatureSelector)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

from classification.common import df_to_xy, read_and_clean_features
from common import get_data_path


def plot_feature_selection(save_loc, file_name, data, ascending):
    df = pd.DataFrame(data)
    df_grp = df[["Feature", "Value"]].groupby("Feature").mean().reset_index()
    df = df.merge(df_grp.rename({"Value":"Mean"}, axis=1), on="Feature")
    df = df.sort_values("Mean", ascending=ascending)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="Feature", y="Value", hue="Measurement",
                ax=ax, palette="Set2")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/fs_{file_name}.png", bbox_inches="tight")
    plt.close()


def univariate(save_loc, X, y, feature_names):
    m_info = mutual_info_classif(X, y)
    m_info = (m_info-m_info.min())/(m_info.max()-m_info.min())
    f_stat, _ = f_classif(X, y)
    f_stat = (f_stat-f_stat.min())/(f_stat.max()-f_stat.min())

    data = []
    for i,name in enumerate(feature_names):
        data.append({"Feature": name,
                     "Measurement": "Mutual Information",
                     "Value":m_info[i]})
        data.append({"Feature": name,
                     "Measurement": "F-Statistic",
                     "Value":f_stat[i]})
    plot_feature_selection(save_loc, "univariate", data, False)


def recursive(save_loc, X, y, feature_names, verbose=False):
    cv = StratifiedKFold(5)
    svm = SVC(kernel="linear")
    svm_ranks = RFECV(svm, cv=cv).fit(X, y)
    rf = RandomForestClassifier()
    rf_ranks = RFECV(rf, cv=cv).fit(X, y)
    lr = LogisticRegression(max_iter=500)
    lr_ranks = RFECV(lr, cv=cv).fit(X, y)
    
    models = {"SVM":svm_ranks,
              "Random Forest":rf_ranks,
              "Logistic Regression":lr_ranks}
    data = []
    for name,model in models.items():
        for i,feature_name in enumerate(feature_names):
            data.append({"Feature": feature_name,
                        "Measurement": name,
                        "Value":model.ranking_[i]})
        if verbose:
            cv_results = pd.DataFrame(model.cv_results_)
            fig, ax = plt.subplots()
            ax.set_xlabel("Number of Features")
            ax.set_ylabel("Mean Test Accuracy")
            ax.errorbar(
                x=cv_results["n_features"],
                y=cv_results["mean_test_score"],
                yerr=cv_results["std_test_score"],
            )
            fig.tight_layout()
            fig.figure.patch.set_alpha(0.0)
            fig.savefig(f"{save_loc}/fs_recursive_{name}.png", bbox_inches="tight")
            plt.close()
    plot_feature_selection(save_loc, "recursive", data, True)


def sfs(X, y, feature_names):
    feature_names = np.array(feature_names)
    cv = StratifiedKFold(5)
    rf = RandomForestClassifier()
    sfs_forward = SequentialFeatureSelector(rf, tol=0.01, direction="forward", cv=cv)
    sfs_forward.fit(X, y)
    print("Forward:", feature_names[sfs_forward.get_support()])
    sfs_backward = SequentialFeatureSelector(rf, tol=-0.01, direction="backward", cv=cv)
    sfs_backward.fit(X, y)
    print("Backward:", feature_names[sfs_backward.get_support()])


def rf_importance(save_loc, X, y, feature_names, n_repeats=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    rf = RandomForestClassifier().fit(X_train, y_train)
    result = permutation_importance(rf, X_test, y_test, n_repeats=n_repeats)

    data = []
    importances = result.importances
    for i,name in enumerate(feature_names):
        for j in range(n_repeats):
            data.append({"Feature": name,
                        "Measurement": "Decrease in Test Accuracy",
                        "Value":importances[i][j]})
    plot_feature_selection(save_loc, "rf", data, False)


def remove_correlated(df, feature_names, label_names):
    feature_names = sorted(list(df.columns))
    [feature_names.remove(ln) for ln in label_names]
    num_features = len(feature_names)

    corr_matrix = df[feature_names].corr()
    features_to_keep = []
    for i in range(num_features):
        feature_i = feature_names[i]
        unique = True
        for j in range(i+1, num_features):
            feature_j = feature_names[j]
            corr = corr_matrix[feature_i][feature_j]
            if abs(corr) > 0.9:
                unique = False
                break
        if unique:
            features_to_keep.append(feature_i)
    print("Non-Correlated:", features_to_keep)


def main(experiment_name, *data_types):
    label = ["game"]
    parent_dir = "."
    if len(data_types[0]) == 1:
        parent_dir = data_types[0][0]
    save_loc = get_data_path(parent_dir, f"model/{experiment_name}/features")
    feature_df = read_and_clean_features(data_types[0], label)
    X, y, _, feature_names = df_to_xy(feature_df, label[0])

    univariate(save_loc, X, y, feature_names)
    recursive(save_loc, X, y, feature_names, True)
    rf_importance(save_loc, X, y, feature_names)
    remove_correlated(feature_df, feature_names, label)
    sfs(X, y, feature_names)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide an experiment name and the data types to train the model with.")
