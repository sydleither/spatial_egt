import muspan as ms
import numpy as np

from data_processing.spatial_statistics.muspan import create_muspan_domain


def create_patches(df, alpha):
    domain = create_muspan_domain(df)
    if len(df[df["type"] == "sensitive"]) > len(df[df["type"] == "resistant"]):
        cell_type = "resistant"
    else:
        cell_type = "sensitive"
    domain.convert_objects(
        population=("type", cell_type),
        collection_name="shape",
        object_type="shape",
        conversion_method="alpha shape",
        conversion_method_kwargs=dict(alpha=alpha)
    )
    return domain


def patch_count(domain):
    patch_pop = ms.query.query(domain, ("collection",), "is", "shape")
    area, _ = ms.geometry.area(domain, population=patch_pop)
    return len(area)


def area_dist(domain):
    patch_pop = ms.query.query(domain, ("collection",), "is", "shape")
    area, _ = ms.geometry.area(domain, population=patch_pop)
    return area


def circularity_dist(domain):
    patch_pop = ms.query.query(domain, ("collection",), "is", "shape")
    circ, _ = ms.geometry.circularity(domain, population=patch_pop)
    return circ


def fractal_dimension_dist(domain):
    patch_pop = ms.query.query(domain, ("collection",), "is", "shape")
    area, _ = ms.geometry.area(domain, population=patch_pop)
    perim, _ = ms.geometry.perimeter(domain, population=patch_pop)
    frac_dim = 2*np.log(np.array(perim))/np.log(np.array(area))
    return frac_dim
