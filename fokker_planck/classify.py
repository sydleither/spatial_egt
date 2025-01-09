import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.optimize import curve_fit

from common.common import game_colors, get_data_path
from common.distributions import get_payoff_data, get_sfp_dist
from fokker_planck.common import (calculate_fp_params, 
                                  classify_game, fokker_planck)


def classify(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = get_payoff_data(processed_data_path)
    df_payoff = df_payoff.apply(calculate_fp_params, axis=1)
    df_payoff = df_payoff.set_index("sample")
    df_payoff["sample"] = df_payoff.index
    sample_ids = list(df_payoff["sample"].values)

    y_pred = []
    y_true = []
    bins = np.linspace(0, 1, 11)
    x_vals = bins[0:-1] + 0.05
    for sample_id in sample_ids[0:1000]:
        sample_df = df_payoff.loc[[sample_id]]
        game = sample_df["game"][0]
        source = sample_df["source"][0]
        sample_dist = get_sfp_dist(processed_data_path, df_payoff, data_type, 
                                   source, sample_id, None, False)
        sample_dist = [1-x for x in sample_dist]
        sample_hist, _ = np.histogram(sample_dist, bins=bins, density=True)
        sample_hist = sample_hist*max(sample_hist)
        try:
            fit_params, _ = curve_fit(fokker_planck, x_vals, sample_hist)
        except:
            fit_params = (0,0,0,0)
        fit_params = tuple(fit_params)
        classification = classify_game(fit_params[1], fit_params[2], fit_params[3])
        y_pred.append(classification)
        y_true.append(game)
    acc = sum([1 for i in range(len(y_pred)) if y_pred[i] == y_true[i]])/len(y_pred)

    games = list(game_colors.keys())
    save_loc = get_data_path(data_type, "images")
    conf_mat = confusion_matrix(y_true, y_pred, labels=games)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=games)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    ax.set_title(f"Accuracy: {acc}")
    fig.tight_layout()
    fig.savefig(f"{save_loc}/fpfit_confusion_matrix.png", bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        classify(sys.argv[1])
    else:
        print("Please provide the data type.")