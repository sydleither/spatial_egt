import sys

from common import get_data_path
from data_analysis.distribution_utils import (get_data, get_data_idv,
                                              plot_agg_line, plot_idv_line)


def idv_plots(data_type, source, *sample_ids):
    save_loc = get_data_path(data_type, "images")
    dists, games = get_data_idv(data_type, source, "rk", sample_ids)
    plot_idv_line(dists, games, save_loc,
                  source+"_rk_"+"_".join(sample_ids),
                  "SR Cross Ripley's K", "r", "cross-K")


def agg_plot(data_type, source):
    save_loc = get_data_path(data_type, "images")
    n = 500
    fs_dists = get_data(data_type, source, "rk", n)
    plot_agg_line(fs_dists, save_loc, source+"_rk",
                  f"SR Cross Ripley's K\n{n} samples",
                  "r", "cross-K")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        agg_plot(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3:
        idv_plots(sys.argv[1], sys.argv[2], *sys.argv[3:])
    else:
        print("Please provide the data type and source.")