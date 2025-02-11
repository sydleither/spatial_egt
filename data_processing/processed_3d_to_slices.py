import os
import sys

import pandas as pd

from common import get_data_path, read_payoff_df


def main(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    slice_data_path = get_data_path(data_type+"_slices", "processed")
    df_payoff = read_payoff_df(processed_data_path)
    new_payoff = []
    for file_name in os.listdir(processed_data_path)[0:5]:
        if file_name == "payoff.csv":
            continue
        source = file_name.split(" ")[0]
        sample = file_name.split(" ")[1][:-4]
        a = df_payoff.at[(source, sample), "a"]
        b = df_payoff.at[(source, sample), "b"]
        c = df_payoff.at[(source, sample), "c"]
        d = df_payoff.at[(source, sample), "d"]
        game = df_payoff.at[(source, sample), "game"]
        df = pd.read_csv(f"{processed_data_path}/{file_name}")
        for slice_dim in ["x", "y", "z"]:
            for slice_val in df[slice_dim].unique():
                df_slice = df[df[slice_dim] == slice_val].drop([slice_dim], axis=1)
                new_source = f"{source}_{sample}"
                new_sample = f"{slice_dim}_{slice_val}"
                new_payoff.append([new_source, new_sample, a, b, c, d, game])
                df_slice.to_csv(f"{slice_data_path}/{new_source} {new_sample}.csv", index=False)
    columns = ["source", "sample", "a", "b","c", "d", "game"]
    new_df_payoff = pd.DataFrame(new_payoff, columns=columns)
    new_df_payoff.to_csv(f"{slice_data_path}/payoff.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the data type.")
