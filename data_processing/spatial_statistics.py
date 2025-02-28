from random import choices
import warnings
warnings.filterwarnings("ignore")

import muspan as ms
import numpy as np
from scipy.stats import skew
from scipy.spatial import KDTree
from trackpy.static import pair_correlation_2d, pair_correlation_3d

from data_processing.ripleyk import calculate_ripley


# Nearest Neighbor
def create_nn_dist(domain, type1_pop, type2_pop):
    nn = ms.spatial_statistics.nearest_neighbour_distribution(
        domain=domain,
        population_A=type1_pop,
        population_B=type2_pop
    )
    return nn


# Cross Pair Correlation Function
def create_cpcf(domain, type1_pop, type2_pop, max_radius, annulus_step, annulus_width):
    _, pcf = ms.spatial_statistics.cross_pair_correlation_function(
        domain=domain,
        population_A=type1_pop,
        population_B=type2_pop,
        max_R=max_radius,
        annulus_step=annulus_step,
        annulus_width=annulus_width
    )
    return pcf


# Pair Correlation Function
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


# Ripley's K
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


def get_muspan_statistics(domain, type1_pop, type2_pop, type1_name, type2_name):
    features = dict()

    anni, _, _ = ms.spatial_statistics.average_nearest_neighbour_index(
        domain=domain,
        population_A=type1_pop,
        population_B=type2_pop
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

    ms.region_based.generate_hexgrid(domain, side_length=2,
                                     regions_collection_name="grids",
                                     remove_empty_regions=False)
    morans_i_s = ms.spatial_statistics.morans_i(domain,
                                              population=("Collection", "grids"),
                                              label_name=f"region counts: {type1_name}")
    features[f"morans_i_{type1_name}"] = morans_i_s[0]
    morans_i_r = ms.spatial_statistics.morans_i(domain,
                                              population=("Collection", "grids"),
                                              label_name=f"region counts: {type2_name}")
    features[f"morans_i_{type2_name}"] = morans_i_r[0]

    wass = ms.distribution.sliced_wasserstein_distance(domain,
                                                       population_A=type1_pop,
                                                       population_B=type2_pop)
    features["wass"] = wass

    kdm_s = ms.distribution.kernel_density_estimation(
        domain,
        population=type1_pop,
        mesh_step=5
    )
    kdm_r = ms.distribution.kernel_density_estimation(
        domain,
        population=type2_pop,
        mesh_step=5
    )
    kl_div = ms.distribution.kl_divergence(kdm_s, kdm_r)
    features["kdm_kl"] = kl_div

    return features


def get_landscape_ecology_features(domain, type_pop, type_name):
    features = dict()

    domain.convert_objects(
        population=type_pop,
        collection_name="shape",
        object_type="shape",
        conversion_method="alpha shape",
        conversion_method_kwargs=dict(alpha=3)
    )
    patch_pop = ms.query.query(domain, ("collection",), "is", "shape")
    circ, _ = ms.geometry.circularity(domain, population=patch_pop)
    area, _ = ms.geometry.area(domain, population=patch_pop)
    perim, _ = ms.geometry.perimeter(domain, population=patch_pop)
    frac_dim = 2*np.log(np.array(perim))/np.log(np.array(area))

    features[f"patches_{type_name}"] = len(circ)
    features = features | get_dist_statistics(f"area_{type_name}", area)
    features = features | get_dist_statistics(f"perim_{type_name}", perim)
    features = features | get_dist_statistics(f"circ_{type_name}", circ)
    features = features | get_dist_statistics(f"fracdim_{type_name}", frac_dim)
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


def get_dist_params(data_type, dimensions):
    dimension = len(dimensions)
    params = {"nc":{}, "sfp":{}, "pcf":{}, "cpcf":{}, "rk":{}}
    if data_type.startswith("in_silico") and dimension == 2:
        params["nc"]["radius"] = 3
        params["sfp"]["sample_length"] = 5
        params["pcf"]["dr"] = 1
        params["pcf"]["max_r"] = 5
        params["cpcf"]["max_radius"] = 5
        params["cpcf"]["annulus_step"] = 1
        params["cpcf"]["annulus_width"] = 3
        params["rk"]["boundary"] = 10
    elif data_type.startswith("in_silico") and dimension == 3:
        params["nc"]["radius"] = 2
        params["sfp"]["sample_length"] = 3
        params["pcf"]["dr"] = 1
        params["pcf"]["max_r"] = 5
        params["cpcf"]["max_radius"] = 3
        params["cpcf"]["annulus_step"] = 1
        params["cpcf"]["annulus_width"] = 1
        params["rk"]["boundary"] = 10
    else:
        params["nc"]["radius"] = 30
        params["sfp"]["sample_length"] = 50
        params["pcf"]["dr"] = 10
        params["pcf"]["max_r"] = 50
        params["cpcf"]["max_radius"] = 50
        params["cpcf"]["annulus_step"] = 10
        params["cpcf"]["annulus_width"] = 30
        params["rk"]["boundary"] = 100
    return params


def create_custom_features(df, data_type, dimensions):
    features = dict()
    s_coords_df = df.loc[df["type"] == "sensitive"][dimensions]
    r_coords_df = df.loc[df["type"] == "resistant"][dimensions]
    s_coords = list(s_coords_df.values)
    r_coords = list(r_coords_df.values)
    params = get_dist_params(data_type, dimensions)
    
    num_sensitive = len(s_coords)
    num_resistant = len(r_coords)
    features["proportion_s"] = num_sensitive/(num_resistant+num_sensitive)

    fs, fr = create_nc_dists(s_coords, r_coords, params["nc"]["radius"])
    features = features | get_dist_statistics("nc_fs", fs)
    features = features | get_dist_statistics("nc_fr", fr)

    pcf_s = create_pcf(s_coords_df, params["pcf"]["max_r"], params["pcf"]["dr"], dimensions)
    pcf_r = create_pcf(r_coords_df, params["pcf"]["max_r"], params["pcf"]["dr"], dimensions)
    features = features | get_dist_statistics("pcf_s", pcf_s)
    features = features | get_dist_statistics("pcf_r", pcf_r)

    rk = create_ripleysk(s_coords, params["rk"]["boundary"], dimensions)
    features = features | get_dist_statistics("rk_s", rk)

    sfp = create_sfp_dist(s_coords, r_coords, params["sfp"]["sample_length"])
    features = features | get_dist_statistics("sfp_fs", sfp)

    return features


def create_muspan_stat_features(df, data_type, dimensions):
    if len(dimensions) == 3:
        print("MuSpan does not support 3D data.")
        exit()

    features = dict()
    domain = create_muspan_domain(df, dimensions)
    s_cells = ms.query.query(domain, ("label", "type"), "is", "sensitive")
    r_cells = ms.query.query(domain, ("label", "type"), "is", "resistant")

    features = features | get_muspan_statistics(domain, s_cells, r_cells,
                                                "sensitive", "resistant")

    return features


def create_muspan_dist_features(df, data_type, dimensions):
    if len(dimensions) == 3:
        print("MuSpan does not support 3D data.")
        exit()

    features = dict()
    params = get_dist_params(data_type, dimensions)
    domain = create_muspan_domain(df, dimensions)
    s_cells = ms.query.query(domain, ("label", "type"), "is", "sensitive")
    r_cells = ms.query.query(domain, ("label", "type"), "is", "resistant")

    pcf = create_cpcf(domain, s_cells, r_cells,
                      params["cpcf"]["max_radius"],
                      params["cpcf"]["annulus_step"],
                      params["cpcf"]["annulus_width"])
    features = features | get_dist_statistics("cpcf", pcf)
    features["cpcf_0"] = pcf[0]

    nn = create_nn_dist(domain, s_cells, r_cells)
    features = features | get_dist_statistics("nn", nn)

    return features


def create_landscape_ecology_featrues(df, data_type, dimensions):
    features = dict()
    domain = create_muspan_domain(df, dimensions)
    s_cells = ms.query.query(domain, ("label", "type"), "is", "sensitive")
    r_cells = ms.query.query(domain, ("label", "type"), "is", "resistant")

    features = features | get_landscape_ecology_features(domain, s_cells, "sensitive")
    features = features | get_landscape_ecology_features(domain, r_cells, "resistant")

    return features
