import os
import sys

import muspan as ms
import pandas as pd
from pandas.api.types import is_object_dtype

from spatial_egt.common import get_data_path
from spatial_database import DOMAIN_STATISTIC_REGISTRY


def main(data_type, domain_name):
    statistics_data_path = get_data_path(data_type, "statistics")
    domain_data_path = f"{statistics_data_path}/{domain_name}"
    for statistic_name, statistic_func in DOMAIN_STATISTIC_REGISTRY.items():
        rows = []
        for domain_file in os.listdir(domain_data_path):
            domain = ms.io.load_domain(f"{domain_data_path}/{domain_file}",
                                       print_metadata=False, print_summary=False)
            statistic = statistic_func(domain)
            source = domain_file.split(" ")[0]
            sample = domain_file.split(" ")[1][:-7]
            rows.append({"source":source, "sample":sample, statistic_name:statistic})
        df = pd.DataFrame(rows)
        if is_object_dtype(df[statistic_name]):
            if statistic_func.__name__.endswith("dist"):
                df["type"] = "distribution"
            else:
                df["type"] = "function"
        else:
            df["type"] = "value"
        df.to_pickle(f"{statistics_data_path}/{statistic_name}.pkl")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the data type and domain calculation function name.")
