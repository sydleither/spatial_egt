import numpy as np
import pandas as pd
from scipy import stats

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


def get_theoretical_dists(n):
    rw_dist = stats.beta.rvs(a=1, b=5, loc=0.1, scale=0.75, size=n)
    co_dist = stats.norm.rvs(loc=0.5, scale=0.2, size=n)
    sw_dist = stats.beta.rvs(a=5, b=1, loc=0.25, scale=0.65, size=n)
    bi_dist = np.concatenate((np.random.choice(rw_dist, n//2), np.random.choice(sw_dist, n//2)), axis=None)
    dists = [sw_dist, co_dist, bi_dist, rw_dist]
    games = ["sensitive_wins", "coexistence", "bistability", "resistant_wins"]
    return dists, games


def fit_beta(sfp_dist):
    bounds = {"a":(0,6), "b":(0,6)}
    res = stats.fit(stats.beta, sfp_dist, bounds=bounds)
    params = res.params
    a = params[0]
    b = params[1]
    loc = params[2]
    scale = params[3]
    return a, b, loc, scale