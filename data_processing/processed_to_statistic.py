"""Convert processed samples into a given spatial statistic.

This script can calculate the spatial statistic for each sample or for individual samples
"""

import argparse
import os

import pandas as pd

from spatial_egt.common import get_data_path
from spatial_database import STATISTIC_PARAMS, STATISTIC_REGISTRY


def calculate_statistics(processed_path, file_names, stat_name, stat_calculation, stat_args):
    """Calulate the spatial statistic of the given sample files"""
    rows = []
    for file_name in file_names:
        try:
            source = file_name.split(" ")[0]
            sample = file_name.split(" ")[1][:-4]
            df_sample = pd.read_csv(f"{processed_path}/{file_name}")
            statistic = stat_calculation(df_sample, **stat_args)
        except Exception as e:
            print(f"Error {file_name}: {e}")
            continue
        rows.append({"source": source, "sample": sample, stat_name: statistic})
    return pd.DataFrame(rows)


def main():
    """Calculate spatial statistic(s) and save as pkl"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--data_type", type=str, default="in_silico")
    parser.add_argument("-time", "--time", type=int, default=72)
    parser.add_argument("-stat", "--statistic", type=str, default=None)
    parser.add_argument("-source", "--source", type=str, default=None)
    parser.add_argument("-sample", "--sample", type=str, default=None)
    args = parser.parse_args()

    processed_path = get_data_path(args.data_type, "processed", args.time)
    stat_name = args.statistic
    print(stat_name)

    stat_calculation = STATISTIC_REGISTRY[stat_name]
    if args.data_type in STATISTIC_PARAMS:
        stat_args_datatype = STATISTIC_PARAMS[args.data_type]
    else:
        stat_args_datatype = STATISTIC_PARAMS
    if stat_name in stat_args_datatype:
        stat_args = stat_args_datatype[stat_name]
    else:
        stat_args = {}

    if args.source is None and args.sample is None:
        statistics_path = get_data_path(args.data_type, "statistics", args.time)
        save_loc = f"{statistics_path}/{stat_name}.pkl"
        file_names = os.listdir(processed_path)
    else:
        statistics_path = get_data_path(args.data_type, f"statistics/{stat_name}", args.time)
        save_loc = f"{statistics_path}/{args.source} {args.sample}.pkl"
        file_names = [f"{args.source} {args.sample}.csv"]

    df = calculate_statistics(
        processed_path, file_names, stat_name, stat_calculation, stat_args
    )
    if len(df) > 0:
        df.to_pickle(save_loc)


if __name__ == "__main__":
    main()
