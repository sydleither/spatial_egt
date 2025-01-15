import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from common.common import game_colors


#features = []
#features = ["nc_fs_mean", "nc_fs_std", "nc_fs_skew"]
#features = ["sfp_fs_mean", "sfp_fs_std", "sfp_fs_skew"]
#features = ["nc_fs_mean", "nc_fs_std", "nc_fs_skew", "nc_fr_mean", "nc_fr_std", "nc_fr_skew"]
#features = ["sfp_fs_mean", "nc_fs_mean", "nc_fr_mean", "nc_mean_diff"]
features = ["nc_fs_mean", "nc_fs_std", "nc_fs_skew", "nc_fr_mean", "nc_fr_std", "nc_fr_skew", "nc_mean_diff", "sfp_fs_mean", "sfp_fs_std", "sfp_fs_skew"]


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


def plot_confusion_matrix(save_loc, file_name, labels, y_test, y_pred, acc):
    conf_mat = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    disp.plot(ax=ax)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    ax.set_title(f"Accuracy: {acc:5.3f}")
    fig.tight_layout()
    fig.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight", transparent=True)
    plt.close()


def plot_prediction_distributions(save_loc, X, feature_names, y_true, y_pred, class_labels, features_to_plot):
    feature_data = list(zip(*X))
    df_dict = {feature_names[i]:feature_data[i] for i in range(len(feature_names))}
    df_dict = df_dict | {"True Label":y_true, "Predicted Label":y_pred}
    df = pd.DataFrame(df_dict)
    df["True Label"] = df["True Label"].map(lambda x: class_labels[x])
    df["Predicted Label"] = df["Predicted Label"].map(lambda x: class_labels[x])
    for feature_name in features_to_plot:
        facet = sns.FacetGrid(df, col="Predicted Label", col_order=class_labels, height=6, aspect=1)
        facet.map_dataframe(sns.histplot, x=feature_name, hue="True Label",
                            palette=game_colors.values(), hue_order=game_colors.keys())
        facet.set_titles(col_template="{col_name}", row_template="{row_name}")
        facet.tight_layout()
        facet.figure.patch.set_alpha(0.0)
        facet.savefig(f"{save_loc}/{feature_name}.png", bbox_inches="tight")