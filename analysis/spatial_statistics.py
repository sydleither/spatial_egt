from csv import reader

# import libpysal as ps
import numpy as np
import pandas as pd
import pointpats as pp
from scipy.stats import skew


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
        features[f"k{dist_name}_mean"] = np.mean(k_dist)
        features[f"k{dist_name}_std"] = np.std(k_dist)
        features[f"k{dist_name}_skew"] = skew(k_dist)
        features[f"k{dist_name}_slope"] = k_dist[-1] - k_dist[0]

    return features


def create_pop_features(num_sensitive, num_resistant):
    features = dict()
    num_total = num_resistant + num_sensitive

    #proportion of cells that are resistant
    features["prop_r"] = num_resistant / num_total

    return features


def create_fs_features(df, num_resistant):
    features = dict()
    df = df.loc[(df["fs"] > 0) & (df["radius"] <= 5) & (df["time"] == df["time"].max())]

    #fs distribution summary statistics
    fs_expand = pd.DataFrame({
        "radius": np.repeat(df["radius"], df["total"]),
        "fs": np.repeat(df["fs"], df["total"])
    })
    agg_funcs = ["mean", "skew", "std", "count"]
    fs_stats = fs_expand.groupby("radius")["fs"].agg(agg_funcs)
    features["fs_mean"] = fs_stats["mean"][3]
    features["fs_skew"] = fs_stats["skew"][3]
    features["fs_std"] = fs_stats["std"][3]

    #slope of mean fs over neighborhood radii
    fs_slope = fs_stats["mean"][5] - fs_stats["mean"][1]
    features["fs_slope"] = fs_slope

    #proportion of R cells that are boundary cells
    r_boundary_prop = fs_stats["count"][1] / num_resistant
    features["r_boundary_prop"] = r_boundary_prop

    return features


# model_state = read_model_state("output/sample10/0/0/2Dmodel250.csv")
# s_coords, r_coords = model_state_to_coords(model_state)
# print(create_ripleysk_features(s_coords, r_coords))