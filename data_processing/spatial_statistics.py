from random import choices
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import skew
from scipy.spatial import KDTree


# Neighborhood Composition Subsample
def create_subnc_dists(s_coords, r_coords, data_type, radius=None, sample_length=None, num_samples=1000):
    s_coords = np.array(s_coords)
    r_coords = np.array(r_coords)
    max_x = max(np.max(s_coords[:, 0]), np.max(r_coords[:, 0]))
    max_y = max(np.max(s_coords[:, 1]), np.max(r_coords[:, 1]))

    if sample_length is None:
        sample_length = 5 if data_type == "in_silico" else 70

    ncsub_agg_fs = []
    ncsub_agg_fr = []
    xs = choices(range(0, max_x-sample_length), k=num_samples)
    ys = choices(range(0, max_y-sample_length), k=num_samples)
    for i in range(num_samples):
        lx = xs[i]
        ux = lx+sample_length
        ly = ys[i]
        uy = ly+sample_length
        subset_s = s_coords[(s_coords[:, 0] >= lx) & (s_coords[:, 0] < ux) & (s_coords[:, 1] >= ly) & (s_coords[:, 1] < uy)]
        subset_r = r_coords[(r_coords[:, 0] >= lx) & (r_coords[:, 0] < ux) & (r_coords[:, 1] >= ly) & (r_coords[:, 1] < uy)]
        sub_fs, sub_fr = create_nc_dists(subset_s, subset_r, data_type, radius=radius)
        if len(sub_fs) != 0:
            ncsub_agg_fs.append(np.mean(sub_fs))
        if len(sub_fr) != 0:
            ncsub_agg_fr.append(np.mean(sub_fr))

    return ncsub_agg_fs, ncsub_agg_fr


# Spatial Subsample
def create_sfp_dist(s_coords, r_coords, data_type, sample_length=None, num_samples=1000, incl_empty=False):
    s_coords = np.array(s_coords)
    r_coords = np.array(r_coords)
    max_x = max(np.max(s_coords[:, 0]), np.max(r_coords[:, 0]))
    max_y = max(np.max(s_coords[:, 1]), np.max(r_coords[:, 1]))

    if sample_length is None:
        sample_length = 5 if data_type == "in_silico" else 70

    fs_counts = []
    xs = choices(range(0, max_x-sample_length), k=num_samples)
    ys = choices(range(0, max_y-sample_length), k=num_samples)
    if incl_empty:
        num_cells = sample_length**2
        for i in range(num_samples):
            lx = xs[i]
            ux = lx+sample_length
            ly = ys[i]
            uy = ly+sample_length
            subset_s = np.sum((s_coords[:, 0] >= lx) & (s_coords[:, 0] < ux) & 
                              (s_coords[:, 1] >= ly) & (s_coords[:, 1] < uy))
            fs_counts.append(subset_s/num_cells)
    else:
        for i in range(num_samples):
            lx = xs[i]
            ux = lx+sample_length
            ly = ys[i]
            uy = ly+sample_length
            subset_s = np.sum((s_coords[:, 0] >= lx) & (s_coords[:, 0] < ux) & 
                              (s_coords[:, 1] >= ly) & (s_coords[:, 1] < uy))
            subset_r = np.sum((r_coords[:, 0] >= lx) & (r_coords[:, 0] < ux) & 
                              (r_coords[:, 1] >= ly) & (r_coords[:, 1] < uy))
            subset_total = subset_s + subset_r
            if subset_total == 0:
                continue
            fs_counts.append(subset_s/subset_total)

    return fs_counts


# Neighborhood Composition
def create_nc_dists(s_coords, r_coords, data_type, radius=None):
    all_coords = np.concatenate((s_coords, r_coords), axis=0)
    if radius is None:
        radius = 3 if data_type == "in_silico" else 30
    s_stop = len(s_coords)
    tree = KDTree(all_coords)
    fs = []
    fr = []
    for p,point in enumerate(all_coords):
        neighbor_indices = tree.query_ball_point(point, radius)
        all_neighbors = len(neighbor_indices)-1
        if p <= s_stop: #sensitive cell
            r_neighbors = len([x for x in neighbor_indices if x > s_stop])
            if all_neighbors != 0 and r_neighbors != 0:
                fr.append(r_neighbors/all_neighbors)
        else: #resistant cell
            s_neighbors = len([x for x in neighbor_indices if x <= s_stop])
            if all_neighbors != 0 and s_neighbors != 0:
                fs.append(s_neighbors/all_neighbors)
    return fs, fr


def get_dist_statistics(name, dist):
    features = dict()
    features[f"{name}_mean"] = np.mean(dist)
    features[f"{name}_std"] = np.std(dist)
    features[f"{name}_skew"] = skew(dist)
    return features


def create_all_features(df, num_sensitive, num_resistant, data_type):
    features = dict()
    s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
    r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)

    features["proportion_s"] = num_sensitive/(num_resistant+num_sensitive)
    fs, fr = create_nc_dists(s_coords, r_coords, data_type)
    features = features | get_dist_statistics("nc_fs", fs)
    features = features | get_dist_statistics("nc_fr", fr)
    sub_fs, sub_fr = create_subnc_dists(s_coords, r_coords, data_type)
    features = features | get_dist_statistics("subnc_fs", sub_fs)
    features = features | get_dist_statistics("subnc_fr", sub_fr)
    sfp = create_sfp_dist(s_coords, r_coords, data_type)
    features = features | get_dist_statistics("sfp_fs", sfp)
    
    return features


def get_cell_type_counts(df):
    counts = df.groupby("type").count().reset_index()
    s = counts[counts["type"] == "sensitive"]["x"].iloc[0]
    r = counts[counts["type"] == "resistant"]["x"].iloc[0]
    return s, r


def calculate_game(payoff):
    a = payoff["a"]
    b = payoff["b"]
    c = payoff["c"]
    d = payoff["d"]
    if a > c and b > d:
        game = "sensitive_wins"
    elif c > a and b > d:
        game = "coexistence"
    elif a > c and d > b:
        game = "bistability"
    elif c > a and d > b:
        game = "resistant_wins"
    else:
        game = "unknown"
    return game


def sample_to_features(df, data_type):
    num_sensitive, num_resistant = get_cell_type_counts(df)
    features = create_all_features(df, num_sensitive, num_resistant, data_type)
    return features