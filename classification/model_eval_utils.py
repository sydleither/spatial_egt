import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import LearningCurveDisplay, StratifiedKFold
import seaborn as sns

from spatial_egt.classification.common import get_model
from spatial_egt.common import game_colors, theme_colors


def plot_performance(save_loc, data_set, int_to_name, y_trues, y_preds):
    plot_confusion_matrix(save_loc, data_set, int_to_name, y_trues, y_preds)
    plot_performance_stats(save_loc, data_set, int_to_name, y_trues, y_preds)


def plt_heatmap(ax, matrix, labels, title, textcolors=("black", "white")):
    '''
    Based on https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    '''
    num_labels = len(labels)
    im = ax.imshow(matrix, cmap="Purples")
    ax.set_xticks(range(num_labels), labels=labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(num_labels), labels=labels)
    threshold = im.norm(matrix.max())/2
    for i in range(num_labels):
        for j in range(num_labels):
            color = textcolors[int(im.norm(matrix[i, j]) > threshold)]
            ax.text(j, i, f"{matrix[i, j]:5.3f}", ha="center", va="center", color=color)
    ax.set(title=title)


def plot_confusion_matrix(save_loc, data_set, int_to_name, y_trues, y_preds):
    labels = [int_to_name[i] for i in range(len(int_to_name))]
    conf_mats = []
    for i in range(len(y_trues)):
        conf_mat = metrics.confusion_matrix(y_trues[i], y_preds[i], normalize="true")
        conf_mats.append(conf_mat)
    conf_mat = np.mean(conf_mats, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    plt_heatmap(ax, conf_mat, labels, "Mean Confusion Matrix")
    ax.set(xlabel="Predicted Game", ylabel="True Game")
    fig.tight_layout()
    fig.savefig(f"{save_loc}/confusion_{data_set}.png", bbox_inches="tight", transparent=True)
    plt.close()


def plot_prediction_distributions(save_loc, data_set, df):
    for feature_name in ["initial_density", "initial_fr", "Stationary Solution"]:
        facet = sns.FacetGrid(df, col="game", col_order=game_colors.keys(), height=6, aspect=1)
        facet.map_dataframe(sns.kdeplot, x=feature_name, hue="correct")
        facet.set_titles(col_template="{col_name}")
        facet.tight_layout()
        facet.figure.patch.set_alpha(0.0)
        facet.savefig(f"{save_loc}/{feature_name}_{data_set}.png", bbox_inches="tight")
        plt.close()


def plot_scatter_prob(save_loc, data_set, df, x, y, hue):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="Purples", ax=ax)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/{x}_{y}_{hue}_{data_set}.png", bbox_inches="tight")
    plt.close()


def get_binary_confusion_matrix(n, y_true, y_pred):
    tp = sum([(y_true[i] == 1) and (y_pred[i] == 1) for i in range(n)])
    fp = sum([(y_true[i] == 0) and (y_pred[i] == 1) for i in range(n)])
    fn = sum([(y_true[i] == 1) and (y_pred[i] == 0) for i in range(n)])
    tn = sum([(y_true[i] == 0) and (y_pred[i] == 0) for i in range(n)])
    return tp, fp, fn, tn


def plot_performance_stats(save_loc, data_set, int_to_name, y_trues, y_preds):
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
            precision = tp/(tp+fp) if (tp+fp) != 0 else 0
            recall = tp/(tp+fn) if (tp+fn) != 0 else 0
            f1 = (2*precision*recall)/(precision+recall) if (precision+recall) != 0 else 0
            df_rows.append({"Game":cat, "Measurement":"Accuracy", "Value":acc, "k":k})
            df_rows.append({"Game":cat, "Measurement":"Precision", "Value":precision, k:"k"})
            df_rows.append({"Game":cat, "Measurement":"Recall/Sensitivity", "Value":recall, "k":k})
            df_rows.append({"Game":cat, "Measurement":"F1 Score", "Value":f1, "k":k})
        overall_accs.append(sum([y_true[i] == y_pred[i] for i in range(n)]) / n)
    mean_acc = np.mean(overall_accs)
    std_acc = np.std(overall_accs)
    df = pd.DataFrame(df_rows)
    
    colors = game_colors.values()
    with sns.axes_style("whitegrid"):
        facet = sns.FacetGrid(df, col="Measurement", height=6, aspect=1)
        facet.map_dataframe(sns.barplot, x="Game", y="Value", hue="Game",
                            errorbar="sd", palette=colors, legend=False)
        facet.set_titles(col_template="{col_name}")
        facet.figure.subplots_adjust(top=0.9)
        facet.figure.suptitle(f"One vs All Statistics\nOverall Accuracy: {mean_acc:5.3f}±{std_acc:5.3f}")
        facet.tight_layout()
        facet.figure.patch.set_alpha(0.0)
        facet.savefig(f"{save_loc}/stats_{data_set}.png", bbox_inches="tight")
        plt.close()


def roc_curve(save_loc, data_set, int_to_name, y_trues, y_probs):
    roc_stats = []
    df_acc_rows = []
    thresholds = np.linspace(0, 1, 101)
    for k in range(len(y_trues)):
        y_true = y_trues[k]
        y_prob = y_probs[k]
        n = len(y_true)
        stats = {"fpr":{}, "tpr":{}}
        for label,cat in int_to_name.items():
            y_label = [1 if y_true[i] == label else 0 for i in range(n)]
            y_pred_prob_label = y_prob[:, label]
            fpr = []
            tpr = []
            for thresh in thresholds:
                y_pred = [1 if y_pred_prob_label[i] > thresh else 0 for i in range(n)]
                tp, fp, fn, tn = get_binary_confusion_matrix(n, y_label, y_pred)
                thresh_fpr = 1-(tn/(fp+tn))
                thresh_tpr = tp/(fn+tp)
                fpr.append(thresh_fpr)
                tpr.append(thresh_tpr)
                acc = (tp+tn)/n
                df_acc_rows.append({"Game":cat, "k":k, "Threshold":thresh, "Accuracy":acc})
            stats["fpr"][cat] = fpr
            stats["tpr"][cat] = tpr
        roc_stats.append(stats)

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    for label,cat in int_to_name.items():
        fprs = []
        tprs = []
        aucs = []
        for stats in roc_stats:
            fpr = stats["fpr"][cat]
            tpr = stats["tpr"][cat]
            ax[label].plot(fpr, tpr, color=game_colors[cat], alpha=0.25)
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(abs(np.trapezoid(y=tpr, x=fpr)))
        avg_fpr = np.mean(np.array(fprs), axis=0)
        avg_tpr = np.mean(np.array(tprs), axis=0)
        avg_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ax[label].plot(avg_fpr, avg_tpr, color=game_colors[cat])
        ax[label].plot(thresholds, thresholds, color="gray", linestyle="--")
        ax[label].set(title=f"{cat}\nAUC: {avg_auc:5.3f}±{std_auc:5.3f}")
    fig.suptitle("One-vs-All ROC Curves")
    fig.supxlabel("False Positive Rate")
    fig.supylabel("True Positive Rate")
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/roc_{data_set}.png", bbox_inches="tight")
    plt.close()

    df_acc = pd.DataFrame(df_acc_rows)
    colors = game_colors.values()
    with sns.axes_style("whitegrid"):
        facet = sns.FacetGrid(df_acc, col="Game", hue="Game",
                              palette=colors, height=6, aspect=1)
        facet.map_dataframe(sns.lineplot, x="Threshold", y="Accuracy", errorbar="sd")
        facet.set_titles(col_template="{col_name}")
        facet.figure.subplots_adjust(top=0.9)
        facet.figure.suptitle("Best Threshold Based on Accuracy")
        facet.tight_layout()
        facet.figure.patch.set_alpha(0.0)
        facet.savefig(f"{save_loc}/rocacc_{data_set}.png", bbox_inches="tight")
        plt.close()


def learning_curve(save_loc, X, y):
    clf = get_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    curves = LearningCurveDisplay.from_estimator(clf, X=X, y=y, cv=cv, ax=ax)
    curves.lines_[0].set_color(theme_colors[0])
    curves.fill_between_[0].set_color(theme_colors[0])
    curves.lines_[1].set_color(theme_colors[1])
    curves.fill_between_[1].set_color(theme_colors[1])
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/learning_curve.png", bbox_inches="tight")
    plt.close()
