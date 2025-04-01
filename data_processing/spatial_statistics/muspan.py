import muspan as ms
import numpy as np


def create_muspan_domain(df):
    domain = ms.domain("sample")
    points = np.asarray([df["x"], df["y"]])
    domain.add_points(points.T, "cells")
    domain.add_labels("type", df["type"])
    return domain


def local_moransi_dist(df, cell_type, side_length):
    domain = create_muspan_domain(df)
    ms.region_based.generate_hexgrid(domain, side_length=side_length,
                                     regions_collection_name="grids",
                                     remove_empty_regions=False)
    _, _, lmi, _, _ = ms.spatial_statistics.morans_i(domain, population=("Collection", "grids"),
                                                     label_name=f"region counts: {cell_type}")
    return lmi


def nn_dist(df, cell_type1="sensitive", cell_type2="resistant"):
    domain = create_muspan_domain(df)
    nn = ms.spatial_statistics.nearest_neighbour_distribution(
        domain=domain,
        population_A=("type", cell_type1),
        population_B=("type", cell_type2)
    )
    return nn


def cpcf(df, max_radius, annulus_step, annulus_width, cell_type1="sensitive", cell_type2="resistant"):
    domain = create_muspan_domain(df)
    _, cpcf = ms.spatial_statistics.cross_pair_correlation_function(
        domain=domain,
        population_A=("type", cell_type1),
        population_B=("type", cell_type2),
        max_R=max_radius,
        annulus_step=annulus_step,
        annulus_width=annulus_width
    )
    return cpcf


def cross_k(df, max_radius, step, cell_type1="sensitive", cell_type2="resistant"):
    domain = create_muspan_domain(df)
    _, ck = ms.spatial_statistics.cross_k_function(
        domain=domain,
        population_A=("type", cell_type1),
        population_B=("type", cell_type2),
        max_R=max_radius,
        step=step
    )
    return ck[1:]


def j_function(df, cell_type, radius_step):
    domain = create_muspan_domain(df)
    _, j, _ = ms.spatial_statistics.J_function(
        domain=domain,
        population=("type", cell_type),
        radius_step=radius_step,
        iterations=10
    )
    return j


def anni(df, cell_type1="sensitive", cell_type2="resistant"):
    domain = create_muspan_domain(df)
    anni, _, _ = ms.spatial_statistics.average_nearest_neighbour_index(
        domain=domain,
        population_A=("type", cell_type1),
        population_B=("type", cell_type2)
    )
    return anni


def entropy(df):
    domain = create_muspan_domain(df)
    entropy = ms.summary_statistics.label_entropy(
        domain=domain,
        label_name="type"
    )
    return entropy


def qcm(df, side_length):
    domain = create_muspan_domain(df)
    ses, _, _ = ms.region_based.quadrat_correlation_matrix(
        domain,
        label_name="type",
        region_method="quadrats",
        region_kwargs=dict(side_length=side_length)
    )
    return ses[0][1]


def global_moransi(df, cell_type, side_length):
    domain = create_muspan_domain(df)
    ms.region_based.generate_hexgrid(domain, side_length=side_length,
                                     regions_collection_name="grids",
                                     remove_empty_regions=False)
    gmi = ms.spatial_statistics.morans_i(domain, population=("Collection", "grids"),
                                         label_name=f"region counts: {cell_type}")
    return gmi[0]


def wasserstein(df, cell_type1="sensitive", cell_type2="resistant"):
    domain = create_muspan_domain(df)
    wass = ms.distribution.sliced_wasserstein_distance(domain,
                                                       population_A=("type", cell_type1),
                                                       population_B=("type", cell_type2))
    return wass


def kl_divergence(df, mesh_step, cell_type1="sensitive", cell_type2="resistant"):
    domain = create_muspan_domain(df)
    kde1 = ms.distribution.kernel_density_estimation(
        domain,
        population=("type", cell_type1),
        mesh_step=mesh_step
    )
    kde2 = ms.distribution.kernel_density_estimation(
        domain,
        population=("type", cell_type2),
        mesh_step=mesh_step
    )
    kl_div = ms.distribution.kl_divergence(kde1, kde2)
    return kl_div
