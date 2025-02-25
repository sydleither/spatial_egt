import random
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import game_colors, get_data_path


def feature_plot(save_loc, feature, df_slices, source_map=None, num_plots=5):
    sources = random.sample(list(df_slices["source"].unique()), num_plots)
    bins = np.arange(-1, 1.01, 0.01)
    fig, ax = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))
    if num_plots == 1:
        ax = [ax]
    for i,source in enumerate(sources):
        df_source = df_slices[df_slices["source"] == source]
        feature_vals = df_source[feature]
        actual = ""
        slice_stats = f"Slice Mean:{feature_vals.mean():5.3f}\nSlice Std:{feature_vals.std():5.3f}"
        if source_map is not None:
            source_val = source_map[source]
            feature_vals = source_val - feature_vals
            actual = f"3D: {source_val:5.3f}\n"
        sns.histplot(np.array(feature_vals), bins=bins, ax=ax[i],
                     stat="density", kde=True, color="#ad8150")
        ax[i].set(xlabel="", ylabel="")
        ax[i].set_xlim(feature_vals.min()-0.01, feature_vals.max()+0.01)
        ax[i].set_title(f"{actual}{slice_stats}")
    if source_map is not None:
        fig.supxlabel("Difference Between Slice Value and 3D Value")
    else:
        fig.supxlabel(feature)
    fig.supylabel("Density")
    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/source_by_{feature}.png", bbox_inches="tight")


def diff_from_actual(save_loc, feature, df_slices, df_3d, group="game"):
    df_3d = df_3d[["uid", "game", feature]]
    df_3d = df_3d.rename({"uid":"source", feature:"3d"}, axis=1)
    df_slices = df_slices[["source", "game", feature]]
    df = df_slices.merge(df_3d, on=["source", "game"])
    df["diff"] = df["3d"] - df[feature]
    df = df[df["game"] != "unknown"]
    
    bins = np.arange(-1, 1.01, 0.01)
    hue_order = None
    palette = None
    if group == "game":
        hue_order=game_colors.keys()
        palette=game_colors.values()
    g = sns.FacetGrid(df, col=group, palette=palette,
                      hue=group, hue_order=hue_order)
    g.map_dataframe(sns.histplot, x="diff", bins=bins, stat="density", kde=True)
    g.set(xlim=(df["diff"].min(), df["diff"].max()))
    g.tight_layout()
    g.figure.patch.set_alpha(0.0)
    g.savefig(f"{save_loc}/diff_in_{feature}_{group}.png", bbox_inches="tight")


def main(feature, slice_dir, source_dir=None):
    slice_data_path = get_data_path(slice_dir, "features")
    df_slices = pd.read_csv(f"{slice_data_path}/all.csv")
    images_data_path = get_data_path(slice_dir, "images")
    source_map = None
    if source_dir is not None:
        source_data_path = get_data_path(source_dir, "features")
        df_3d = pd.read_csv(f"{source_data_path}/all.csv")
        df_3d["uid"] = df_3d["source"]+"_"+df_3d["sample"].astype(str)
        source_map = df_3d[["uid", feature]].set_index("uid").to_dict()[feature]
        diff_from_actual(images_data_path, feature, df_slices, df_3d)
    feature_plot(images_data_path, feature, df_slices, source_map)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the feature name, slice directory name, and (optional) 3d directory name.")
