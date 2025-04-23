"""Convert processed samples into a given spatial statistic.

This script can calculate the spatial statistic for each sample or for individual samples

Expected usage:
python3 -m spatial_egt.data_processing.processed_to_statistic
    data_type stat_name (source) (sample)

Where:
data_type: the name of the directory in data/ containing the processed/ data
stat_name: the name of the spatial statistic to calculate, as defined in spatial_database.py
source: optional
    the source of the sample to calculate individually
sample: optional
    the sample id to calculate individually
"""

import os
import sys

import pandas as pd

from spatial_egt.common import get_data_path
from spatial_database import STATISTIC_PARAMS, STATISTIC_REGISTRY


def calculate_statistics(processed_path, file_names, stat_name, stat_calculation, stat_args):
    """Calulate the spatial statistic of the given sample files

    :param processed_path: path to the processed data
    :type processed_path: str
    :param file_names: the processed data files
    :type file_names: list[str]
    :param stat_name: name of spatial statistic to calculate
    :type stat_name: str
    :param stat_calculation: the function that calculates the spatial statistic
    :type stat_calculation: function
    :param stat_args: arguments for calculation
    :type stat_args: dict
    :return: a dataframe containing the source, sample, and calculated statistic
    :rtype: Pandas DataFrame
    """
    rows = []
    for file_name in file_names:
        source = file_name.split(" ")[0]
        sample = file_name.split(" ")[1][:-4]
        print(f"{source} {sample}")
        df_sample = pd.read_csv(f"{processed_path}/{file_name}")
        statistic = stat_calculation(df_sample, **stat_args)
        rows.append({"source": source, "sample": sample, stat_name: statistic})
    return pd.DataFrame(rows)


def main(data_type, stat_name, source=None, sample=None):
    """Calculate spatial statistic(s) and save as pkl"""
    processed_path = get_data_path(data_type, "processed")

    stat_calculation = STATISTIC_REGISTRY[stat_name]
    if data_type in STATISTIC_PARAMS:
        stat_args_datatype = STATISTIC_PARAMS[data_type]
    else:
        stat_args_datatype = STATISTIC_PARAMS
    if stat_name in stat_args_datatype:
        stat_args = stat_args_datatype[stat_name]
    else:
        stat_args = {}

    if source is None and sample is None:
        statistics_path = get_data_path(data_type, "statistics")
        save_loc = f"{statistics_path}/{stat_name}.pkl"
        file_names = os.listdir(processed_path)
    else:
        statistics_path = get_data_path(data_type, f"statistics/{stat_name}")
        save_loc = f"{statistics_path}/{source} {sample}.pkl"
        file_names = [f"{source} {sample}.csv"]

    df = calculate_statistics(
        processed_path, file_names, stat_name, stat_calculation, stat_args
    )
    df.to_pickle(save_loc)


if __name__ == "__main__":
    if len(sys.argv) in (3, 5):
        main(*sys.argv[1:])
    else:
        print("Please see the module docstring for usage instructions.")
