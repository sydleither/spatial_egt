from collections import Counter, OrderedDict
from random import sample
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import game_colors, get_data_path
from data_processing.processed_to_features import read_processed_sample
from data_processing.spatial_statistics import (calculate_game, create_sfp_dist)


def fx(x, awm, amw, s, mu):
    if awm == 0:
        return(s*x)
    f = (-x + mu*np.log(1-x)+mu*np.log(x)-x*amw/awm + np.log(1+x*awm)*(amw + (1+s+amw)*awm)/(awm**2))
    return f
    

def potential(x, n, mu, a12, a21, s):
    phi = np.log(x*(1-x)/2*n) - 2*n*fx(x, a12, a21, s, mu)
    return phi


def get_sfp_dist(processed_data_path, df_payoff, data_type, source, sample_id, subset_size, incl_empty):
    file_name = f"spatial_{source}_{sample_id}.csv"
    df = read_processed_sample(processed_data_path, file_name, df_payoff)
    s_coords = list(df.loc[df["type"] == "sensitive"][["x", "y"]].values)
    r_coords = list(df.loc[df["type"] == "resistant"][["x", "y"]].values)
    dist = create_sfp_dist(s_coords, r_coords, data_type, subset_size, 1000, incl_empty)
    return dist


def calculate_fp_params(row):
    epsilon = 1 - row["a"]
    awm = row["b"] + epsilon - 1
    sm = row["d"] + epsilon - 1
    amw = row["c"] + epsilon - 1 - sm
    row["awm"] = awm
    row["sm"] = sm
    row["amw"] = amw
    return row


def main(data_type, source, *sample_ids):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff = df_payoff[df_payoff["source"] == source]
    df_payoff["sample"] = df_payoff["sample"].astype(str)
    df_payoff["game"] = df_payoff.apply(calculate_game, axis="columns")
    df_payoff = df_payoff[df_payoff["game"] != "unknown"]
    df_payoff = df_payoff.apply(calculate_fp_params, axis=1)
    df_payoff = df_payoff.set_index("sample")
    df_payoff["sample"] = df_payoff.index

    if not sample_ids:
        sample_ids = sample(list(df_payoff.index.values), 4)

    n = 1000
    p = np.linspace(0.01, 0.99, n)
    mu = 0.01
    sfp_supports = []
    sfp_freqs = []
    fp_probs = []
    games = []
    for sample_id in sample_ids:
        sample_data = df_payoff.loc[[sample_id]]
        games.append(sample_data["game"][0])
        sample_dist = get_sfp_dist(processed_data_path, df_payoff,
                                   data_type, source, sample_id, 7, False)
        sample_dist = [round(x, 1) for x in sample_dist]
        counts = Counter(sample_dist)
        counts = OrderedDict(sorted(counts.items()))
        y_sum = sum(counts.values())
        freqs = [y/y_sum for y in counts.values()]
        sfp_supports.append(counts.keys())
        sfp_freqs.append(freqs)
        q = potential(p, n, mu, sample_data["awm"][0], 
                      sample_data["amw"][0], sample_data["sm"][0])
        fp_probs.append(np.exp(-q))
    
    save_loc = get_data_path(data_type, "images")
    num_samples = len(sample_ids)
    fig, ax = plt.subplots(1, num_samples, figsize=(6*num_samples, 5))
    if num_samples == 1:
        ax = [ax]
    for i in range(num_samples):
        fp_prob = fp_probs[i]
        ax[i].bar(sfp_supports[i], sfp_freqs[i], 
                  width=0.1, color=game_colors[games[i]])
        ax[i].plot(p, fp_prob/max(fp_prob), color="black", linewidth=2)
        ax[i].set(xlim=(0,1), ylim=(0,1))
    # fig.supxlabel("Fraction Mutant")
    # fig.supylabel("Probability Density")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    file_name = source+"_fp_"+"_".join(sample_ids)
    fig.savefig(f"{save_loc}/{file_name}.png")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], *sys.argv[3:])
    else:
        print("Please provide the data type and source.")