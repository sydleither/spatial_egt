import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LearningCurveDisplay, StratifiedKFold
import seaborn as sns

from classification.common import get_model
from common import game_colors


def plot_all(save_loc, int_to_name, y_trues, y_preds, data_type):
    plot_confusion_matrix(save_loc, data_type, int_to_name, y_trues, y_preds)
    plot_performance_stats(save_loc, f"stats_{data_type}", int_to_name, y_trues, y_preds)


def plt_heatmap(ax, matrix, labels, title):
    '''
    Based on https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    '''
    num_labels = len(labels)
    ax.imshow(matrix, cmap="Greens")
    ax.set_xticks(range(num_labels), labels=labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(num_labels), labels=labels)
    for i in range(num_labels):
        for j in range(num_labels):
            ax.text(j, i, f"{matrix[i, j]:5.3f}", ha="center", va="center", color="deeppink")
    ax.set(title=title)


def plot_confusion_matrix(save_loc, data_type, int_to_name, y_trues, y_preds):
    labels = [int_to_name[i] for i in range(len(int_to_name))]
    conf_mats = []
    for i in range(len(y_trues)):
        conf_mat = confusion_matrix(y_trues[i], y_preds[i], normalize="true")
        conf_mats.append(conf_mat)
    avg_conf_mat = np.mean(conf_mats, axis=0)
    std_conf_mat = np.std(conf_mats, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt_heatmap(ax[0], avg_conf_mat, labels, "Confusion Matrix Mean")
    plt_heatmap(ax[1], std_conf_mat, labels, "Confusion Matrix Std")
    fig.tight_layout()
    fig.savefig(f"{save_loc}/confusion_{data_type}.png", bbox_inches="tight", transparent=True)
    plt.close()


# def plot_prediction_distributions(save_loc, X, feature_names, y_true, y_pred, int_to_name, features_to_plot):
#     feature_data = list(zip(*X))
#     df_dict = {feature_names[i]:feature_data[i] for i in range(len(feature_names))}
#     df_dict = df_dict | {"True Label":y_true, "Predicted Label":y_pred}
#     df = pd.DataFrame(df_dict)
#     df["True Label"] = df["True Label"].map(lambda x: int_to_name[x])
#     df["Predicted Label"] = df["Predicted Label"].map(lambda x: int_to_name[x])
#     for feature_name in features_to_plot:
#         facet = sns.FacetGrid(df, col="Predicted Label", col_order=game_colors.keys(), height=6, aspect=1)
#         facet.map_dataframe(sns.histplot, x=feature_name, hue="True Label",
#                             palette=game_colors.values(), hue_order=game_colors.keys())
#         facet.set_titles(col_template="{col_name}", row_template="{row_name}")
#         facet.tight_layout()
#         facet.figure.patch.set_alpha(0.0)
#         facet.savefig(f"{save_loc}/{feature_name}.png", bbox_inches="tight")
#         plt.close()


def get_binary_confusion_matrix(n, y_true, y_pred):
    tp = sum([(y_true[i] == 1) and (y_pred[i] == 1) for i in range(n)])
    fp = sum([(y_true[i] == 0) and (y_pred[i] == 1) for i in range(n)])
    fn = sum([(y_true[i] == 1) and (y_pred[i] == 0) for i in range(n)])
    tn = sum([(y_true[i] == 0) and (y_pred[i] == 0) for i in range(n)])
    return tp, fp, fn, tn


def plot_performance_stats(save_loc, file_name, int_to_name, y_trues, y_preds):
    overall_accs = []
    df_rows = []
    for k in range(len(y_trues)):
        y_true = y_trues[k]
        y_pred = y_preds[k]
        n = len(y_true)
        for label,cat in int_to_name.items():
            y_true_label = [1 if y_true[i] == label else 0 for i in range(n)]
            y_pred_label = [1 if y_pred[i] == label else 0 for i in range(n)]
            tp, fp, fn, tn = get_binary_confusion_matrix(n, y_true_label, y_pred_label)
            acc = (tp+tn)/n
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1 = (2*precision*recall)/(precision+recall)
            df_rows.append({"Game":cat, "Measurement":"Accuracy", "Value":acc, "k":k})
            df_rows.append({"Game":cat, "Measurement":"Precision", "Value":precision, k:"k"})
            df_rows.append({"Game":cat, "Measurement":"Recall/Sensitivity", "Value":recall, "k":k})
            df_rows.append({"Game":cat, "Measurement":"F1 Score", "Value":f1, "k":k})
        overall_accs.append(sum([y_true[i] == y_pred[i] for i in range(n)]) / n)
    mean_acc = np.mean(overall_accs)
    std_acc = np.std(overall_accs)
    
    colors = game_colors.values()
    df = pd.DataFrame(df_rows)
    with sns.axes_style("whitegrid"):
        facet = sns.FacetGrid(df, col="Measurement", height=6, aspect=1)
        facet.map_dataframe(sns.barplot, x="Game", y="Value", hue="Game",
                            errorbar="sd", palette=colors, legend=False)
        facet.set_titles(col_template="{col_name}")
        facet.figure.subplots_adjust(top=0.9)
        facet.figure.suptitle(f"One vs All Statistics\nOverall Accuracy: {mean_acc:5.3f}±{std_acc:5.3f}")
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


def learning_curve(save_loc, X, y):
    clf = get_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    LearningCurveDisplay.from_estimator(clf, X=X, y=y, cv=cv, ax=ax)
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/learning_curve.png", bbox_inches="tight")
    plt.close()