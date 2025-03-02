from random import choices

import numpy as np
from scipy.spatial import KDTree
# from trackpy.static import pair_correlation_2d, pair_correlation_3d

# from data_processing.spatial_statistics.ripleyk import calculate_ripley


# def pcf(df, max_r, dr):
#     dimensions = list(df.drop("type", axis=1).columns)
#     if "z" in dimensions and len(dimensions) == 2:
#         if "x" in df.columns:
#             df = df.rename({"z":"y"}, axis=1)
#         else:
#             df = df.rename({"z":"x"}, axis=1)
#     if len(dimensions) == 2:
#         _, gr = pair_correlation_2d(df, max_r, dr=dr, fraction=0.5)
#     else:
#         _, gr = pair_correlation_3d(df, max_r, dr=dr, fraction=0.5)
#     return gr


# def ripleysk(df, cell_type, bounding_radius):
#     dimensions = list(df.drop("type", axis=1).columns)
#     coords = df[df["type"] == cell_type][dimensions]
#     d1 = coords[:, 0]
#     d2 = coords[:, 1]
#     d3 = None
#     if len(dimensions) == 3:
#         d3 = coords[:, 2]
#     radii = np.arange(0, bounding_radius+1, 1).tolist()
#     k = calculate_ripley(radii, 10, d1=d1, d2=d2, d3=d3, CSR_Normalise=True)
#     return k


# Spatial Subsample
def sfp_dist(df, sample_length, num_samples=1000):
    dimensions = list(df.drop("type", axis=1).columns)
    s_coords = df[df["type"] == "sensitive"][dimensions].values
    r_coords = df[df["type"] == "resistant"][dimensions].values
    dims = range(len(dimensions))

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
def nc_dist(df, radius, return_fs=True):
    dimensions = list(df.drop("type", axis=1).columns)
    s_coords = df[df["type"] == "sensitive"][dimensions].values
    r_coords = df[df["type"] == "resistant"][dimensions].values
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

    if return_fs:
        return fs
    return fr


def proportion_s(df):
    num_sensitive = len(df[df["type"] == "sensitive"])
    num_resistant = len(df[df["type"] == "resistant"])
    return num_sensitive/(num_resistant+num_sensitive)
