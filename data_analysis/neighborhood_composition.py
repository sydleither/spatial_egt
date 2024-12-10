import sys

from common import get_data_path
from data_analysis.distribution_utils import (get_data, get_data_idv, plot_agg_dist, 
                                              plot_idv_fs_count, plot_idv_dist)


def idv_plots(data_type, source, *sample_ids):
    save_loc = get_data_path(data_type, "images")
    dists, games = get_data_idv(data_type, source, "nc", sample_ids)
    plot_idv_dist(dists, games, save_loc, source+"_nc_"+"_".join(sample_ids),
                  "Neighborhood Composition Distributions", "Fraction Sensitive",
                  "Fraction of Resistant", 1)
    plot_idv_fs_count(dists, games, save_loc,
                      source+"_ncraw_"+"_".join(sample_ids),
                      "Fraction Sensitive Across Subsamples",
                      "Resistant Cell", "Fraction Sensitive")


def agg_plot(data_type, source):
    save_loc = get_data_path(data_type, "images")
    n = 500
    fs_dists = get_data(data_type, source, "nc", n)
    plot_agg_dist(fs_dists, save_loc, source+"_nc",
                  f"Neighborhood Composition Distributions\n{n} samples",
                  "Fraction Sensitive", "Fraction of Resistant", 1)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        agg_plot(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3:
        idv_plots(sys.argv[1], sys.argv[2], *sys.argv[3:])
    else:
        print("Please provide the data type and source.")