"""Combine individual sample's spatial statistics into one pkl.

For use when processed_to_statistics was run over individual samples.

Expected usage: python3 -m spatial_egt.data_processing.combine_sample_statistics
    data_type statistic_name time

where:
data_type: the name of the directory in data/
statisitic_name: the name of directory in data/{data_type}/statisitcs
    that holds the individual samples
time: timepoint
"""

import os
import sys

import pandas as pd

from spatial_egt.common import get_data_path


def main(data_type, statistic_name, time):
    """Combine individual sample's spatial statistics"""
    save_loc = get_data_path(data_type, "statistics", time)
    statistics_path = f"{save_loc}/{statistic_name}"
    df = pd.DataFrame()
    for file_name in os.listdir(statistics_path):
        df_sample = pd.read_pickle(f"{statistics_path}/{file_name}")
        df = pd.concat([df, df_sample])
    df.to_pickle(f"{save_loc}/{statistic_name}.pkl")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(*sys.argv[1:])
    else:
        print("Please see the module docstring for usage instructions.")
