import os
import sys

import pandas as pd

from common import get_data_path


def main(feature_set, data_type):
    data_path = get_data_path(data_type, f"model/{feature_set}/model_iteration")
    df = pd.DataFrame()
    for file in os.listdir(data_path):
        df_fs = pd.read_csv(f"{data_path}/{file}")
        cols = [x for x in df_fs.columns if x != "mean"]
        df_fs["features"] = df_fs[cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
        df_fs["number"] = int(file[:-4])
        df = pd.concat([df, df_fs[["number", "features", "mean"]]])

    for number in df["number"].unique():
        df_num = df[df["number"] == number]
        print(df_num.nlargest(3, "mean").to_string())


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide a feature set and the data type.")
