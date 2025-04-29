import sys

import pandas as pd

from spatial_egt.common import get_data_path


def main(data_type, label_name):
    features_data_path = get_data_path(data_type, "statistics")
    df = pd.read_csv(f"{features_data_path}/features.csv")

    total_samples = len(df)
    print(f"Total Samples: {total_samples}")
    for label in df[label_name].unique():
        df_label = df[df[label_name] == label]
        print(f"\t{label}: {len(df_label)}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        print("Please provide the data type and label name.")
