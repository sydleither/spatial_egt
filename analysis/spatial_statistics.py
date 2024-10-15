from csv import reader

import numpy as np
import pandas as pd
import pointpats as pp
from scipy.stats import skew
from scipy.spatial import Voronoi


def read_model_state(file_loc):
    model_state = list(reader(open(file_loc)))
    model_state = [[cell for cell in row[0]] for row in model_state]
    return model_state


def model_state_to_coords(model_state):
    s_coords = []
    r_coords = []
    for x in range(len(model_state)):
        for y in range(len(model_state[0])):
            if model_state[x][y] == "S":
                s_coords.append((x,y))
            elif model_state[x][y] == "R":
                r_coords.append((x,y))
    return s_coords, r_coords


def create_voroni_features(s_coords, r_coords):
    features = dict()

    s_vor = Voronoi(s_coords)
    r_vor = Voronoi(r_coords)

    features["vor_s"] = len(s_vor.regions)
    features["vor_r"] = len(r_vor.regions)

    return features


def create_ripleysk_features(s_coords, r_coords):
    features = dict()

    s_k = pp.l(np.array(s_coords), support=(2,6,5))[1]
    r_k = pp.l(np.array(r_coords), support=(2,6,5))[1]
    all_k = pp.l(np.array(s_coords+r_coords), support=(2,6,5))[1]

    k_dists = [s_k, r_k, all_k]
    k_dists_names = ["s", "r", "all"]
    for i in range(len(k_dists)):
        dist_name = k_dists_names[i]
        k_dist = k_dists[i]
        features[f"k_{dist_name}_mean"] = np.mean(k_dist)
        features[f"k_{dist_name}_std"] = np.std(k_dist)
        features[f"k_{dist_name}_skew"] = skew(k_dist)
        features[f"k_{dist_name}_slope"] = k_dist[-1] - k_dist[0]

    return features


def create_pop_features(num_sensitive, num_resistant):
    features = dict()
    num_total = num_resistant + num_sensitive

    #proportion of cells that are resistant
    features["prop_r"] = num_resistant / num_total

    return features


def create_pc_features(df, num_sensitive, num_resistant):
    features = dict()
    df["pc"] = df["normalized_count"] / (num_sensitive*num_resistant)
    df = df.loc[(df["radius"] <= 5) & (df["time"] == df["time"].max())]

    #pc distribution summary statistics
    pairs = df["pair"].unique()
    for pair in pairs:
        pair_dist = df.loc[df["pair"] == pair]["pc"].values
        features[f"pc_{pair}_mean"] = np.mean(pair_dist)
        features[f"pc_{pair}_std"] = np.std(pair_dist)
        features[f"pc_{pair}_skew"] = skew(pair_dist)
        features[f"pc_{pair}_slope"] = pair_dist[-1] - pair_dist[0]

    return features


def create_fsfr_features(df_fs, df_fr, num_resistant, num_sensitive):
    features = dict()

    for fi in ["fr", "fs"]:
        if fi == "fr":
            df = df_fr.loc[(df_fr[fi] > 0) & (df_fr["radius"] <= 5) & (df_fr["time"] == df_fr["time"].max())]
        else:
            df = df_fs.loc[(df_fs[fi] > 0) & (df_fs["radius"] <= 5) & (df_fs["time"] == df_fs["time"].max())]

        #fs distribution summary statistics
        fs_expand = pd.DataFrame({
            "radius": np.repeat(df["radius"], df["total"]),
            fi: np.repeat(df[fi], df["total"])
        })
        agg_funcs = ["mean", "skew", "std", "count"]
        fs_stats = fs_expand.groupby("radius")[fi].agg(agg_funcs)
        features[f"{fi}_mean"] = fs_stats["mean"][3]
        features[f"{fi}_skew"] = fs_stats["skew"][3]
        features[f"{fi}_std"] = fs_stats["std"][3]

        #slope of mean fs over neighborhood radii
        fs_slope = fs_stats["mean"][5] - fs_stats["mean"][1]
        features[f"{fi}_slope"] = fs_slope

        #proportion of R cells that are boundary cells
        num_i = num_resistant if fi == "fs" else num_sensitive
        r_boundary_prop = fs_stats["count"][1] / num_i
        features[f"{fi[-1]}_boundary_prop"] = r_boundary_prop

    return features