import numpy as np
import pointpats as pp
from scipy.stats import skew
from scipy.spatial import KDTree
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


def create_fp_features(s_coords, r_coords):
    features = dict()
    all_coords = [("s", s_coords[i][0], s_coords[i][1]) for i in range(len(s_coords))]
    all_coords += [("r", r_coords[i][0], r_coords[i][1]) for i in range(len(r_coords))]

    max_x = max([x[1] for x in all_coords])
    max_y = max([x[2] for x in all_coords])
    max_coord = max(max_x, max_y)

    fs_counts = []
    subset_size = 10
    for s in range(max_coord//subset_size):
        lower = s*subset_size
        upper = (s+1)*subset_size
        subset = [t for t,x,y in all_coords if lower <= x <= upper and lower <= y <= upper]
        subset_total = len(subset)
        subset_s = len([x for x in subset if x[0] == "s"])
        if subset_total == 0:
            continue
        fs_counts.append(subset_s/subset_total)
    
    features["fp_fs_mean"] = np.mean(fs_counts)
    features["fp_fs_std"] = np.std(fs_counts)
    features["fp_fs_skew"] = skew(fs_counts)
    return features


def create_frfs_features(s_coords, r_coords):
    features = dict()
    all_coords = s_coords + r_coords
    radius = 3

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

    features["fs_mean"] = np.mean(fs)
    features["fs_std"] = np.std(fs)
    features["fs_skew"] = skew(fs)
    features["fr_mean"] = np.mean(fr)
    features["fr_std"] = np.std(fr)
    features["fr_skew"] = skew(fr)
    return features


def create_all_features(df, num_sensitive, num_resistant):
    features = dict()
    s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
    r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)

    features["proportion_s"] = num_sensitive/(num_resistant+num_sensitive)
    features = features | create_frfs_features(s_coords, r_coords)
    features = features | create_fp_features(s_coords, r_coords)
    
    return features


def get_cell_type_counts(df):
    counts = df.groupby("type").count().reset_index()
    s = counts[counts["type"] == "sensitive"]["x"].iloc[0]
    r = counts[counts["type"] == "resistant"]["x"].iloc[0]
    return s, r


def calculate_game(payoff):
    a = payoff["a"].iloc[0]
    b = payoff["b"].iloc[0]
    c = payoff["c"].iloc[0]
    d = payoff["d"].iloc[0]
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


def process_sample(df):
    num_sensitive, num_resistant = get_cell_type_counts(df)
    features = create_all_features(df, num_sensitive, num_resistant)
    return features