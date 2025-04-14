import os
import sys

from pandas.api.types import is_object_dtype
import pandas as pd

from spatial_egt.common import get_data_path
from spatial_database import STATISTIC_PARAMS, STATISTIC_REGISTRY


def calculate_statistics(processed_path, file_names, statistic_name, statistic_calculation, statistic_args):
    rows = []
    for file_name in file_names:
        if file_name == "payoff.csv":
            continue
        df_sample = pd.read_csv(f"{processed_path}/{file_name}")
        try:
            statistic = statistic_calculation(df_sample, **statistic_args)
        except Exception as e:
            print(f"Error: {file_name}")
            print(e)
            continue
        source = file_name.split(" ")[0]
        sample = file_name.split(" ")[1][:-4]
        rows.append({"source": source, "sample": sample, statistic_name: statistic})
    return rows


def main(data_type, statistic_name, source=None, sample=None):
    processed_path = get_data_path(data_type, "processed")

    statistic_calculation = STATISTIC_REGISTRY[statistic_name]
    statistic_args_datatype = STATISTIC_PARAMS[data_type]
    statistic_args = dict()
    if statistic_name in statistic_args_datatype:
        statistic_args = statistic_args_datatype[statistic_name]

    if source is None and sample is None:
        statistics_path = get_data_path(data_type, "statistics")
        save_loc = f"{statistics_path}/{statistic_name}.pkl"
        file_names = os.listdir(processed_path)
    else:
        print(source, sample)
        statistics_path = get_data_path(data_type, f"statistics/{statistic_name}")
        save_loc = f"{statistics_path}/{source} {sample}.pkl"
        file_names = [f"{source} {sample}.csv"]

    rows = calculate_statistics(
        processed_path, file_names, statistic_name, statistic_calculation, statistic_args
    )
    df = pd.DataFrame(rows)
    if is_object_dtype(df[statistic_name]):
        if statistic_calculation.__name__.endswith("dist"):
            df["type"] = "distribution"
        else:
            df["type"] = "function"
    else:
        df["type"] = "value"
    df.to_pickle(save_loc)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the data type and statistic name")
        print("and a source and sample id, if calculating individual samples.")
