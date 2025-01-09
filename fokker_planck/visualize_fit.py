from random import sample
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from common.common import game_colors, get_data_path
from common.distributions import get_payoff_data, get_sfp_dist
from fokker_planck.common import (calculate_fp_params, 
                                  classify_game, fokker_planck)


def main(data_type, source, *sample_ids):
    processed_data_path = get_data_path(data_type, "processed")
    save_loc = get_data_path(data_type, "images")
    df_payoff = get_payoff_data(processed_data_path)
    df_payoff = df_payoff.apply(calculate_fp_params, axis=1)
    df_payoff = df_payoff.set_index("sample")
    df_payoff["sample"] = df_payoff.index

    if not sample_ids:
        sample_ids = sample(list(df_payoff.index.values), 4)

    bins = np.linspace(0, 1, 11)
    x_vals = bins[0:-1] + 0.05
    num_samples = len(sample_ids)
    fig, ax = plt.subplots(1, num_samples, figsize=(5*num_samples, 4))
    if num_samples == 1:
        ax = [ax]
    for i in range(len(sample_ids)):
        sample_id = sample_ids[i]
        sample_df = df_payoff.loc[[sample_id]]
        game = sample_df["game"][0]
        sample_dist = get_sfp_dist(processed_data_path, df_payoff, data_type, 
                                   source, sample_id, None, False)
        sample_dist = [1-x for x in sample_dist]
        sample_hist, _ = np.histogram(sample_dist, bins=bins, density=True)
        sample_hist = sample_hist*max(sample_hist)
        try:
            fit_params, _ = curve_fit(fokker_planck, x_vals, sample_hist)
        except:
            fit_params = (0,0,0,0)
        true_params = (0, sample_df["awm"][0], sample_df["amw"][0], sample_df["sm"][0])
        fit_game = classify_game(fit_params[1], fit_params[2], fit_params[3])
        ax[i].plot(x_vals, sample_hist/max(sample_hist), c=game_colors[game], linewidth=2)
        pred_hist = fokker_planck(x_vals, *fit_params)
        ax[i].plot(x_vals, pred_hist/max(pred_hist), c=game_colors[fit_game], ls="--", linewidth=2)
        pred = "pred: mu=%5.2f, awm=%5.2f, amw=%5.2f, sm=%5.2f" % tuple(fit_params)
        true = "true: mu=%5.2f, awm=%5.2f, amw=%5.2f, sm=%5.2f" % tuple(true_params)
        ax[i].set(title=f"{pred}\n{true}")
    fig.supxlabel("Fraction Mutant/Resistant")
    fig.supylabel("Probability Density")
    fig.tight_layout()
    fig.patch.set_alpha(0)
    file_name = source+"_fpfit_"+"_".join(sample_ids)
    fig.savefig(f"{save_loc}/{file_name}.png", bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], *sys.argv[3:])
    else:
        print("Please provide the data type and source.")