import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import skew
from scipy.spatial import KDTree


# Spatial Fokker Planck
def create_sfp_dist(s_coords, r_coords, subset_size=4, incl_empty=False):
    all_coords = [("s", s_coords[i][0], s_coords[i][1]) for i in range(len(s_coords))]
    all_coords += [("r", r_coords[i][0], r_coords[i][1]) for i in range(len(r_coords))]

    max_x = max([x[1] for x in all_coords])
    max_y = max([x[2] for x in all_coords])
    max_coord = max(max_x, max_y)

    fs_counts = []
    num_subsets = max_coord//subset_size
    if incl_empty:
        num_cells = subset_size**2
        for s in range(num_subsets):
            lower = s*subset_size
            upper = (s+1)*subset_size
            subset = [t for t,x,y in all_coords if lower <= x < upper and lower <= y < upper]
            subset_s = len([x for x in subset if x[0] == "s"])
            fs_counts.append(subset_s/num_cells)
    else:
        for s in range(num_subsets):
            lower = s*subset_size
            upper = (s+1)*subset_size
            subset = [t for t,x,y in all_coords if lower <= x < upper and lower <= y < upper]
            subset_total = len(subset)
            if subset_total == 0:
                continue
            subset_s = len([x for x in subset if x[0] == "s"])
            fs_counts.append(subset_s/subset_total)

    return fs_counts


# Neighborhood Composition
def create_nc_dists(s_coords, r_coords, data_type):
    all_coords = s_coords + r_coords
    radius = 3 if data_type == "in_silico" else 10
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
    sfp = create_sfp_dist(s_coords, r_coords)
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