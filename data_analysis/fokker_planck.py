from random import sample
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common.common import game_colors, get_data_path
from common.distributions import get_sfp_dist
from data_processing.spatial_statistics import calculate_game


def fx(x, awm, amw, s, mu):
    if awm == 0:
        return(s*x)
    f = (-x + mu*np.log(1-x)+mu*np.log(x)-x*amw/awm + np.log(1+x*awm)*(amw + (1+s+amw)*awm)/(awm**2))
    return f


def potential(x, n, mu, awm, amw, s):
    phi = np.log(x*(1-x)/2*n) - 2*n*fx(x, awm, amw, s, mu)
    return phi


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

    p_fp = np.linspace(0.01, 0.99, 1000)
    p_sfp = np.linspace(0.01, 0.99, 10)
    mu = 0.01
    sfp_probs = []
    fp_probs = []
    games = []
    for sample_id in sample_ids:
        sample_data = df_payoff.loc[[sample_id]]
        games.append(sample_data["game"][0])
        sample_dist = get_sfp_dist(processed_data_path, df_payoff,
                                   data_type, source, sample_id, 7, False)
        sample_dist = [1-x for x in sample_dist]
        hist, _ = np.histogram(sample_dist, bins=p_sfp, density=True)
        sfp_probs.append(hist)
        q = potential(p_fp, 1000, mu, sample_data["awm"][0], 
                      sample_data["amw"][0], sample_data["sm"][0])
        fp_probs.append(np.exp(-q))
    
    save_loc = get_data_path(data_type, "images")
    num_samples = len(sample_ids)
    fig, ax = plt.subplots(1, num_samples, figsize=(6*num_samples, 5))
    if num_samples == 1:
        ax = [ax]
    for i in range(num_samples):
        fp_prob = fp_probs[i]
        sfp_prob = sfp_probs[i]
        ax[i].stairs(sfp_prob/max(sfp_prob), p_sfp, 
                     color=game_colors[games[i]], fill=True)
        ax[i].plot(p_fp, fp_prob/max(fp_prob), color="black", linewidth=3)
        ax[i].set(xlim=(0,1), ylim=(0,1))
    fig.supxlabel("Fraction Mutant/Resistant")
    fig.supylabel("Probability Density")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    file_name = source+"_fp_"+"_".join(sample_ids)
    fig.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], *sys.argv[3:])
    else:
        print("Please provide the data type and source.")