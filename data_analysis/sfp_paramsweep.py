from random import sample
import sys

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

from common import get_data_path
from data_processing.processed_to_features import read_processed_sample
from data_processing.spatial_statistics import (calculate_game,
                                                create_sfp_dist)


def get_sfp_dist(processed_data_path, df_payoff, sample_id, subset_size, incl_empty):
    file_name = f"spatial_HAL_{sample_id}.csv"
    df = read_processed_sample(processed_data_path,
                               file_name, df_payoff)
    s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
    r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)
    dist = create_sfp_dist(s_coords, r_coords, subset_size, incl_empty)
    return dist


def main(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    df_payoff["game"] = df_payoff.apply(calculate_game, axis="columns")
    coexist_ids = list(df_payoff[df_payoff["game"] == "coexistence"]["sample"].values)
    bistability_ids = list(df_payoff[df_payoff["game"] == "bistability"]["sample"].values)
    subset_sizes = list(range(4, 22, 2))
    results = {s:[] for s in subset_sizes}
    for i in range(1000):
        co_sample = sample(coexist_ids, 1)[0]
        bi_sample = sample(bistability_ids, 1)[0]
        subset_size = sample(subset_sizes, 1)[0]
        co_dist = get_sfp_dist(processed_data_path, df_payoff, 
                               co_sample, subset_size, False)
        bi_dist = get_sfp_dist(processed_data_path, df_payoff, 
                               bi_sample, subset_size, False)
        _, p = mannwhitneyu(co_dist, bi_dist)
        results[subset_size].append(p<0.005)
    
    save_loc = get_data_path(data_type, "images")
    prop_sig = {k:sum(v)/len(v) for k,v in results.items()}
    fig, ax = plt.subplots()
    ax.bar(x=prop_sig.keys(), height=prop_sig.values(), color="forestgreen")
    ax.set(xlabel="subset size", ylabel="proportion significant")
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/sfp_paramsweep.png")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the data type.")