import sys

import muspan as ms
import pandas as pd

from common import get_data_path
from data_processing.feature_database import DOMAIN_PARAMS, DOMAIN_REGISTRY


def main(data_type, feature_name, source, sample):
    print(source, sample)
    processed_path = get_data_path(data_type, "processed")

    feature_calculation = DOMAIN_REGISTRY[feature_name]
    feature_args_datatype = DOMAIN_PARAMS[data_type]
    feature_args = dict()
    if feature_name in feature_args_datatype:
        feature_args = feature_args_datatype[feature_name]

    df_sample = pd.read_csv(f"{processed_path}/{source} {sample}.csv")
    domain = feature_calculation(df_sample, **feature_args)
    features_path = get_data_path(data_type, f"features/{feature_name}")
    ms.io.save_domain(domain, path_to_save=features_path, name_of_file=f"{source} {sample}")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Please provide the data type, domain calculation function name, source and sample id.")
