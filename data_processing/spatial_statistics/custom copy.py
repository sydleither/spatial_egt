from random import choices

import numpy as np
from scipy.spatial import KDTree
from trackpy.static import pair_correlation_2d, pair_correlation_3d

from data_processing.spatial_statistics.ripleyk import calculate_ripley


# Pair Correlation Function (3D)
def create_pcf(df, max_r, dr, dimensions):
    dimension = len(dimensions)
    if "z" in df.columns and dimension == 2:
        if "x" in df.columns:
            df = df.rename({"z":"y"}, axis=1)
        else:
            df = df.rename({"z":"x"}, axis=1)
    if dimension == 2:
        _, gr = pair_correlation_2d(df, max_r, dr=dr, fraction=0.5)
    else:
        _, gr = pair_correlation_3d(df, max_r, dr=dr, fraction=0.5)
    return gr


# Ripley's K (3D)
def create_ripleysk(coords, bounding_radius, dimensions):
    coords = np.array(coords)
    d1 = coords[:, 0]
    d2 = coords[:, 1]
    d3 = None
    if len(dimensions) == 3:
        d3 = coords[:, 2]
    radii = np.arange(0, bounding_radius+1, 1).tolist()
    k = calculate_ripley(radii, 10, d1=d1, d2=d2, d3=d3, CSR_Normalise=True)
    return k


# Spatial Subsample
def create_sfp_dist(s_coords, r_coords, sample_length, num_samples=1000):
    dims = range(len(s_coords[0]))
    s_coords = np.array(s_coords)
    r_coords = np.array(r_coords)
    max_dims = [max(np.max(s_coords[:, i]), np.max(r_coords[:, i])) for i in dims]
    dim_vals = [choices(range(0, max_dims[i]-sample_length), k=num_samples) for i in dims]
    fs_counts = []
    for s in range(num_samples):
        ld = [dim_vals[i][s] for i in dims]
        ud = [ld[i]+sample_length for i in dims]
        subset_s = [(s_coords[:, i] >= ld[i]) & (s_coords[:, i] <= ud[i]) for i in dims]
        subset_s = np.sum(np.all(subset_s, axis=0))
        subset_r = [(r_coords[:, i] >= ld[i]) & (r_coords[:, i] <= ud[i]) for i in dims]
        subset_r = np.sum(np.all(subset_r, axis=0))
        subset_total = subset_s + subset_r
        if subset_total == 0:
            continue
        fs_counts.append(subset_s/subset_total)
    return fs_counts


# Neighborhood Composition
def create_nc_dists(s_coords, r_coords, radius):
    all_coords = np.concatenate((s_coords, r_coords), axis=0)
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


def create_custom_features(df, data_type):
    features = dict()

    dimensions = list(df.drop("type", axis=1).columns)
    s_coords_df = df.loc[df["type"] == "sensitive"][dimensions]
    r_coords_df = df.loc[df["type"] == "resistant"][dimensions]
    s_coords = list(s_coords_df.values)
    r_coords = list(r_coords_df.values)
    params = get_dist_params(data_type, dimensions)
    
    num_sensitive = len(s_coords)
    num_resistant = len(r_coords)
    features["Proportion Sensitive"] = num_sensitive/(num_resistant+num_sensitive)

    fs, fr = create_nc_dists(s_coords, r_coords, params["nc"]["radius"])
    features = features | get_dist_statistics("NC (Resistant)", fs)
    features = features | get_dist_statistics("NC (Sensitive)", fr)

    sfp = create_sfp_dist(s_coords, r_coords, params["sfp"]["sample_length"])
    features = features | get_dist_statistics("SFP", sfp)

    return features
