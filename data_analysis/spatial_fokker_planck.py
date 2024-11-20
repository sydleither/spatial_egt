import sys

from common import get_data_path
from data_analysis.distribution_utils import (get_data, plot_agg_dist, 
                                              plot_idv_dist)


def main(data_type):
    save_loc = get_data_path(data_type, "images")
    n = 500
    fs_dists = get_data(data_type, "sfp", n)
    plot_agg_dist(fs_dists, save_loc, "spatial_fokker_planck",
                  f"Spatial Fokker-Planck Distributions\n{n} samples",
                  "Fraction Sensitive", "Frequency Across Subsamples", 1)
    plot_idv_dist(fs_dists, save_loc, "spatial_fokker_planck_idv",
                  f"Spatial Fokker-Planck Distributions",
                  "Fraction Sensitive", "Frequency Across Subsamples", 1)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the data type.")