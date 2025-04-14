import sys

import muspan as ms
import pandas as pd

from spatial_egt.common import get_data_path
from spatial_database import DOMAIN_PARAMS, DOMAIN_REGISTRY


def main(data_type, statistic_name, source, sample):
    print(source, sample)
    processed_path = get_data_path(data_type, "processed")

    statistic_calculation = DOMAIN_REGISTRY[statistic_name]
    statistic_args_datatype = DOMAIN_PARAMS[data_type]
    statistic_args = {}
    if statistic_name in statistic_args_datatype:
        statistic_args = statistic_args_datatype[statistic_name]

    df_sample = pd.read_csv(f"{processed_path}/{source} {sample}.csv")
    domain = statistic_calculation(df_sample, **statistic_args)
    statistics_path = get_data_path(data_type, f"statistics/{statistic_name}")
    ms.io.save_domain(domain, path_to_save=statistics_path, name_of_file=f"{source} {sample}")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Please provide the data type, domain calculation function name, source and sample id.")
