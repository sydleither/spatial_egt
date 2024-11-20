import sys

from common import get_data_path
from data_analysis.distribution_utils import get_data, plot_agg_dist


def main(data_type):
    save_loc = get_data_path(data_type, "images")
    n = 100
    all_fs = get_data(data_type, "nc", 100)
    plot_agg_dist(all_fs, save_loc, "neighborhood_composition",
              f"Neighborhood Composition Distributions\n{n} samples",
              "Fraction Sensitive in Radius 3 from Resistant", 
              "Proportion of Resistant", 1)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the data type.")