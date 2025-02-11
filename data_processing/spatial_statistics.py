from random import choices
import warnings
warnings.filterwarnings("ignore")

import muspan as ms
import numpy as np
from scipy.stats import skew
from scipy.spatial import KDTree


# Nearest Neighbor
def create_nn_dist(domain, type1, type2):
    population_A = ms.query.query(domain, ("label", "type"), "is", type1)
    population_B = ms.query.query(domain, ("label", "type"), "is", type2)
    nn = ms.spatial_statistics.nearest_neighbour_distribution(
        domain=domain,
        population_A=population_A,
        population_B=population_B
    )

    return nn.tolist()


# Cross Pair Correlation Function
def create_cpcf(domain, type1, type2, max_radius, annulus_step, annulus_width):
    population_A = ms.query.query(domain, ("label", "type"), "is", type1)
    population_B = ms.query.query(domain, ("label", "type"), "is", type2)
    _, pcf = ms.spatial_statistics.cross_pair_correlation_function(
        domain=domain,
        population_A=population_A,
        population_B=population_B,
        max_R=max_radius,
        annulus_step=annulus_step,
        annulus_width=annulus_width
    )

    return pcf


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


def get_muspan_statistics(domain):
    features = dict()
    s_cells = ms.query.query(domain, ("label", "type"), "is", "sensitive")
    r_cells = ms.query.query(domain, ("label", "type"), "is", "resistant")

    anni, _, _ = ms.spatial_statistics.average_nearest_neighbour_index(
        domain=domain,
        population_A=s_cells,
        population_B=r_cells
    )
    features["anni"] = anni

    densities, _, labels = ms.summary_statistics.label_density(
        domain=domain,
        label_name="type"
    )
    features[f"{labels[0]}_density"] = densities[0]
    features[f"{labels[1]}_density"] = densities[1]

    entropy = ms.summary_statistics.label_entropy(
        domain=domain,
        label_name="type"
    )
    features["entropy"] = entropy

    ses3, _, _ = ms.region_based.quadrat_correlation_matrix(
        domain,
        label_name="type",
        region_method="quadrats",
        region_kwargs=dict(side_length=3)
    )
    features["ses3"] = ses3[0][1]

    ses5, _, _ = ms.region_based.quadrat_correlation_matrix(
        domain,
        label_name="type",
        region_method="quadrats",
        region_kwargs=dict(side_length=5)
    )
    features["ses5"] = ses5[0][1]

    return features


def get_dist_statistics(name, dist):
    features = dict()
    features[f"{name}_mean"] = np.mean(dist)
    features[f"{name}_std"] = np.std(dist)
    features[f"{name}_skew"] = skew(dist)
    return features


def create_muspan_domain(df, dimensions):
    domain = ms.domain("sample")
    points = np.asarray([df[x] for x in dimensions])
    domain.add_points(points.T, "cells")
    domain.add_labels("type", df["type"])
    return domain


def get_dist_params(data_type):
    params = {"nc":{}, "sfp":{}, "cpfc":{}}
    if data_type.startswith("in_silico"):
        params["nc"]["radius"] = 3
        params["sfp"]["sample_length"] = 5
        params["cpfc"]["max_radius"] = 5
        params["cpfc"]["annulus_step"] = 1
        params["cpfc"]["annulus_width"] = 3
    else:
        params["nc"]["radius"] = 30
        params["sfp"]["sample_length"] = 50
        params["pcf"]["max_radius"] = 50
        params["pcf"]["annulus_step"] = 10
        params["pcf"]["annulus_width"] = 30
    return params


def create_custom_features(df, data_type, dimensions):
    features = dict()
    s_coords = list(df.loc[df["type"] == "sensitive"][dimensions].values)
    r_coords = list(df.loc[df["type"] == "resistant"][dimensions].values)
    params = get_dist_params(data_type)
    
    num_sensitive = len(s_coords)
    num_resistant = len(r_coords)
    features["proportion_s"] = num_sensitive/(num_resistant+num_sensitive)

    fs, fr = create_nc_dists(s_coords, r_coords, params["nc"]["radius"])
    features = features | get_dist_statistics("nc_fs", fs)
    features = features | get_dist_statistics("nc_fr", fr)

    sfp = create_sfp_dist(s_coords, r_coords, params["sfp"]["sample_length"])
    features = features | get_dist_statistics("sfp_fs", sfp)

    return features


def create_muspan_features(df, data_type, dimensions):
    if len(dimensions) == 3:
        print("MuSpan does not support 3D data.")
        exit()

    features = dict()
    params = get_dist_params(data_type)

    domain = create_muspan_domain(df, dimensions)
    features = features | get_muspan_statistics(domain)

    pcf = create_cpcf(domain, "sensitive", "resistant",
                      params["pcf"]["max_radius"],
                      params["pcf"]["annulus_step"],
                      params["pcf"]["annulus_width"])
    features = features | get_dist_statistics("pcf", pcf)
    features["pcf_0"] = pcf[0]

    nn = create_nn_dist(domain, "sensitive", "resistant")
    features = features | get_dist_statistics("nn", nn)

    return features
