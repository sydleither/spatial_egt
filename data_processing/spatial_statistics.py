from csv import reader

import numpy as np
import pandas as pd
import pointpats as pp
from scipy.stats import skew
from scipy.spatial import Voronoi


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


# def create_frfs_features(s_coords, r_coords):
#     features = dict()
#     all_coords = [(i,"s",s_coords[i]) for i in range(len(s_coords))]
#     all_coords += [(i+len(s_coords),"r",r_coords[i]) for i in range(len(r_coords))]

#     radius = 3
#     neighbors = dict()
#     for c1_idx,c1_type,c1_coords in all_coords:
#         neighbors[c1_idx] = (0,0)
#         for c2_idx,c2_type,c2_coords in all_coords:
#             if abs(s_cell[0] - r_cell[0]) < radius and abs(s_cell[1] - r_cell[1]) < radius:
#                 neighbors[s_cell][0] += 1


def create_all_features(df, num_sensitive, num_resistant):
    features = dict()
    s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
    r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)

    features = features | {"proportion_r": num_resistant/(num_resistant+num_sensitive)}
    features = features | create_ripleysk_features(s_coords, r_coords)
    features = features | create_voroni_features(s_coords, r_coords)
    
    return features


def get_cell_type_counts(df):
    counts = df.groupby("type").count().reset_index()
    s = counts[counts["type"] == "sensitive"]["x"].values[0]
    r = counts[counts["type"] == "resistant"]["x"].values[0]
    return s,r


def process_sample(df):
    num_sensitive, num_resistant = get_cell_type_counts(df)
    features = create_all_features(df, num_sensitive, num_resistant)
    print(features)