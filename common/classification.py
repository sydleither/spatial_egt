import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from common.common import game_colors


#features = []
#features = ["nc_fs_mean", "nc_fs_std", "nc_fs_skew"]
#features = ["sfp_fs_mean", "sfp_fs_std", "sfp_fs_skew"]
features = ["nc_fs_mean", "nc_fs_std", "nc_fs_skew", "nc_fr_mean", "nc_fr_std", "nc_fr_skew"]
#features = ["subnc_fs_mean", "subnc_fs_std", "subnc_fs_skew", "subnc_fr_mean", "subnc_fr_std", "subnc_fr_skew"]


def clean_feature_data(df):
    df = df[df["game"] != "unknown"]
    df = df[df["proportion_s"] <= 0.95]
    df = df[df["proportion_s"] >= 0.05]
    skew_features = [x for x in df.columns if "skew" in x]
    for feature in skew_features:
        df[feature].fillna(0)
    df = df.dropna()
    return df


def df_to_xy(df):
    label_name = "game"
    feature_names = list(df.columns)
    feature_names.remove(label_name)
    label_categories = list(game_colors.keys())
    category_to_int = {lc:i for i,lc in enumerate(label_categories)}
    int_to_category = {i:lc for i,lc in enumerate(label_categories)}
    X = list(df[feature_names].values)
    y = [category_to_int[x] for x in df[label_name].values]
    return X, y, int_to_category


def plot_confusion_matrix(save_loc, file_name, int_to_name, y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [int_to_name[i] for i in range(len(int_to_name))]
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    disp.plot(ax=ax)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight", transparent=True)
    plt.close()


def plot_prediction_distributions(save_loc, X, feature_names, y_true, y_pred, int_to_name, features_to_plot):
    feature_data = list(zip(*X))
    df_dict = {feature_names[i]:feature_data[i] for i in range(len(feature_names))}
    df_dict = df_dict | {"True Label":y_true, "Predicted Label":y_pred}
    df = pd.DataFrame(df_dict)
    df["True Label"] = df["True Label"].map(lambda x: int_to_name[x])
    df["Predicted Label"] = df["Predicted Label"].map(lambda x: int_to_name[x])
    for feature_name in features_to_plot:
        facet = sns.FacetGrid(df, col="Predicted Label", col_order=game_colors.keys(), height=6, aspect=1)
        facet.map_dataframe(sns.histplot, x=feature_name, hue="True Label",
                            palette=game_colors.values(), hue_order=game_colors.keys())
        facet.set_titles(col_template="{col_name}", row_template="{row_name}")
        facet.tight_layout()
        facet.figure.patch.set_alpha(0.0)
        facet.savefig(f"{save_loc}/{feature_name}.png", bbox_inches="tight")
        plt.close()


def get_binary_confusion_matrix(n, y_true, y_pred):
    tp = sum([(y_true[i] == 1) and (y_pred[i] == 1) for i in range(n)])
    fp = sum([(y_true[i] == 0) and (y_pred[i] == 1) for i in range(n)])
    fn = sum([(y_true[i] == 1) and (y_pred[i] == 0) for i in range(n)])
    tn = sum([(y_true[i] == 0) and (y_pred[i] == 0) for i in range(n)])
    return tp, fp, fn, tn


def plot_performance_stats(save_loc, file_name, int_to_name, y_true, y_pred):
    n = len(y_true)
    df_rows = []
    for label,cat in int_to_name.items():
        y_true_label = [1 if y_true[i] == label else 0 for i in range(n)]
        y_pred_label = [1 if y_pred[i] == label else 0 for i in range(n)]
        tp, fp, fn, tn = get_binary_confusion_matrix(n, y_true_label, y_pred_label)
        acc = (tp+tn)/n
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = (2*precision*recall)/(precision+recall)
        df_rows.append({"Game":cat, "Measurement":"Accuracy", "Value":acc})
        df_rows.append({"Game":cat, "Measurement":"Precision", "Value":precision})
        df_rows.append({"Game":cat, "Measurement":"Recall/Sensitivity", "Value":recall})
        df_rows.append({"Game":cat, "Measurement":"F1 Score", "Value":f1})
    overall_acc = sum([y_true[i] == y_pred[i] for i in range(n)]) / n
    
    df = pd.DataFrame(df_rows)
    with sns.axes_style("whitegrid"):
        facet = sns.FacetGrid(df, col="Measurement", height=6, aspect=1)
        facet.map_dataframe(sns.barplot, x="Game", y="Value", hue="Game",
                            palette=game_colors.values(), legend=False)
        facet.set_titles(col_template="{col_name}")
        facet.figure.subplots_adjust(top=0.9)
        facet.figure.suptitle(f"Overall Accuracy: {overall_acc:5.3f}")
        facet.tight_layout()
        facet.figure.patch.set_alpha(0.0)
        facet.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")
        plt.close()


def roc_curve(save_loc, file_name, int_to_name, clf, X, y):
    y_pred_prob = clf.predict_proba(X)

    n = len(y)
    thresholds = np.linspace(0, 1, 101)
    step = thresholds[1]
    stats = {"fpr":{}, "tpr":{}, "auc":{}, "acc":{}}
    for label,cat in int_to_name.items():
        y_label = [1 if y[i] == label else 0 for i in range(n)]
        y_pred_prob_label = y_pred_prob[:, label]
        fpr = []
        tpr = []
        auc = 0
        acc = []
        for thresh in thresholds:
            y_pred = [1 if y_pred_prob_label[i] > thresh else 0 for i in range(n)]
            tp, fp, fn, tn = get_binary_confusion_matrix(n, y_label, y_pred)
            thresh_fpr = 1-(tn/(fp+tn))
            thresh_tpr = tp/(fn+tp)
            fpr.append(thresh_fpr)
            tpr.append(thresh_tpr)
            auc += thresh_tpr*step + (step*thresh_fpr)/2
            acc.append((tp+tn)/n)
        stats["fpr"][cat] = fpr
        stats["tpr"][cat] = tpr
        stats["auc"][cat] = 2*auc-1
        stats["acc"][cat] = acc

    fig, ax = plt.subplots(figsize=(6,5))
    for cat in int_to_name.values():
        ax.plot(stats["fpr"][cat], stats["tpr"][cat],
                color=game_colors[cat], label=f"{cat}: {stats['auc'][cat]:5.3f}")
    ax.plot(thresholds, thresholds, color="gray", linestyle="--")
    ax.set(title="One-vs-Rest ROC Curves", xlabel="False Positive Rate", ylabel="True Positive Rate")
    fig.legend(loc="center right")
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(1, len(int_to_name), figsize=(5*len(int_to_name),5))
    for i,cat in enumerate(int_to_name.values()):
        ax[i].bar(thresholds, stats["acc"][cat], width=step, color=game_colors[cat])
        ax[i].set(title=cat)
    fig.suptitle("Best Threshold Based on Accuracy")
    fig.supxlabel("Threshold")
    fig.supylabel("Accuracy")
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/acc_{file_name}.png", bbox_inches="tight")
    plt.close()
