import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (f_classif,
                                       mutual_info_classif,
                                       RFECV)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from classification.common import df_to_xy, get_model, read_and_clean_features
from classification.performance_plots import plot_all, learning_curve, roc_curve
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
    lr = LogisticRegression()
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


def main(experiment_name, *data_types):
    label = ["game"]
    parent_dir = "."
    if len(data_types[0]) == 1:
        parent_dir = data_types[0][0]
    save_loc = get_data_path(parent_dir, f"model/{experiment_name}")
    feature_df = read_and_clean_features(data_types[0], label)
    X, y, _, feature_names = df_to_xy(feature_df, label[0])

    univariate(save_loc, X, y, feature_names)
    recursive(save_loc, X, y, feature_names, True)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide an experiment name and the data types to train the model with.")
