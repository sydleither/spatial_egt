from random import choices
import sys

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

from common import get_data_path
from data_processing.processed_to_features import read_processed_sample
from data_processing.spatial_statistics import calculate_game, create_sfp_dist


def get_sfp_dist(processed_data_path, df_payoff, data_type, source, sample_id, subset_size, incl_empty):
    file_name = f"spatial_{source}_{sample_id}.csv"
    df = read_processed_sample(processed_data_path, file_name, df_payoff)
    s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
    r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)
    dist = create_sfp_dist(s_coords, r_coords, data_type, subset_size, 1000, incl_empty)
    return dist


def get_payoff_data(processed_data_path):
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    df_payoff["game"] = df_payoff.apply(calculate_game, axis="columns")
    df_payoff = df_payoff[df_payoff["game"] != "unknown"]
    return df_payoff


def main(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    df_payoff["game"] = df_payoff.apply(calculate_game, axis="columns")

    num_samples = 5000
    subset_sizes = range(2, 11) if data_type == "in_silico" else range(10, 110, 10)
    coexist_ids = list(df_payoff[df_payoff["game"] == "coexistence"][["sample", "source"]].values)
    bistability_ids = list(df_payoff[df_payoff["game"] == "bistability"][["sample", "source"]].values)
    subset_samples = choices(subset_sizes, k=num_samples)
    coexist_samples = choices(coexist_ids, k=num_samples)
    bistability_samples = choices(bistability_ids, k=num_samples)

    results = []
    for i in range(num_samples):
        co_sample, co_source = coexist_samples[i]
        bi_sample, bi_source = bistability_samples[i]
        subset_size = subset_samples[i]
        co_dist = get_sfp_dist(processed_data_path, df_payoff, data_type,
                               co_source, co_sample, subset_size, False)
        bi_dist = get_sfp_dist(processed_data_path, df_payoff, data_type,
                               bi_source, bi_sample, subset_size, False)
        score = stats.wasserstein_distance(co_dist, bi_dist)
        results.append([subset_size, score])

    save_loc = get_data_path(data_type, "images")
    name = "Earth Movers Distance"
    df = pd.DataFrame(data=results, columns=["Subset Size", name])
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Subset Size", y=name, color="#cf6275", ax=ax)
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(f"{save_loc}/sfp_paramsweep.png")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the data type.")