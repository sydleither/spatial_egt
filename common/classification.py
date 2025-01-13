import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from common.common import game_colors


#features = []
#features = ["nc_fs_mean", "nc_fs_std", "nc_fs_skew"]
#features = ["sfp_fs_mean", "sfp_fs_std", "sfp_fs_skew"]
#features = ["nc_fs_mean", "nc_fs_std", "nc_fs_skew", "nc_fr_mean", "nc_fr_std", "nc_fr_skew"]
features = ["sfp_fs_mean", "nc_fs_mean", "nc_fr_mean", "nc_mean_diff"]


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


def plot_confusion_matrix(save_loc, title, labels, y_test, y_pred, acc):
    conf_mat = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    disp.plot(ax=ax)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    ax.set_title(f"Accuracy: {acc:5.3f}")
    fig.tight_layout()
    fig.savefig(f"{save_loc}/{title}.png", bbox_inches="tight", transparent=True)
    plt.close()