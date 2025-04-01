import os
import sys

import muspan as ms
import pandas as pd
from pandas.api.types import is_object_dtype

from common import get_data_path
from data_processing.feature_database import DOMAIN_FEATURES


def main(data_type, domain_name):
    features_data_path = get_data_path(data_type, f"features")
    domain_data_path = f"{features_data_path}/{domain_name}"
    for feature_name, feature_func in DOMAIN_FEATURES.items():
        rows = []
        for domain_file in os.listdir(domain_data_path):
            domain = ms.io.load_domain(f"{domain_data_path}/{domain_file}",
                                       print_metadata=False, print_summary=False)
            feature = feature_func(domain)
            source = domain_file.split(" ")[0]
            sample = domain_file.split(" ")[1][:-7]
            rows.append({"source":source, "sample":sample, feature_name:feature})
        df = pd.DataFrame(rows)
        if is_object_dtype(df[feature_name]):
            if feature_func.__name__.endswith("dist"):
                df["type"] = "distribution"
            else:
                df["type"] = "function"
        else:
            df["type"] = "value"
        df.to_pickle(f"{features_data_path}/{feature_name}.pkl")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the data type and domain calculation function name.")
