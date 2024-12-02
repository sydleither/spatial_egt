import sys

from common import get_data_path
from data_analysis.distribution_utils import (get_data, get_data_idv, plot_agg_dist, 
                                              plot_idv_fs_count, plot_idv_dist)


def idv_plots(data_type,source, *sample_ids):
    save_loc = get_data_path(data_type, "images")
    dists, games = get_data_idv(data_type, source, "sfp", sample_ids)
    plot_idv_dist(dists, games, save_loc, source+"_sfp_"+"_".join(sample_ids),
                  "Spatial Fokker-Planck Distributions", "Fraction Sensitive",
                  "Frequency Across Subsamples", 1)
    plot_idv_fs_count(dists, games, save_loc,
                      "sfpraw_"+"_".join(sample_ids),
                      "Fraction Sensitive Across Subsamples",
                      "Subsample", "Fraction Sensitive")


def agg_plot(data_type, source):
    save_loc = get_data_path(data_type, "images")
    n = 100
    fs_dists = get_data(data_type, source, "sfp", n)
    plot_agg_dist(fs_dists, save_loc, source+"_sfp",
                  f"Spatial Fokker-Planck Distributions\n{n} samples",
                  "Fraction Sensitive", "Frequency Across Subsamples", 1)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        agg_plot(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 2:
        idv_plots(sys.argv[1], sys.argv[2], *sys.argv[3:])
    else:
        print("Please provide the data type and source.")