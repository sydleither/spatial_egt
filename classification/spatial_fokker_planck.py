from random import sample
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.optimize import curve_fit

from common.common import game_colors, get_data_path
from common.distributions import get_payoff_data, get_sfp_dist


def fokker_planck(x, mu, awm, amw, s):
    n = 10
    if awm == 0:
        fx = s*x
    else:
        fx = (-x + mu*np.log(1-x)+mu*np.log(x)-x*amw/awm +
              np.log(1+x*awm)*(amw + (1+s+amw)*awm)/(awm**2))
    phi = np.log(x*(1-x)/2*n) - 2*n*fx
    return np.exp(-phi)


def calculate_fp_params(row):
    epsilon = 1 - row["a"]
    awm = row["b"] + epsilon - 1
    sm = row["d"] + epsilon - 1
    amw = row["c"] + epsilon - 1 - sm
    row["awm"] = awm
    row["sm"] = sm
    row["amw"] = amw
    return row


def classify_game(awm, amw, sm):
    a = 0
    b = awm
    c = sm+amw
    d = sm
    if a > c and b > d:
        game = "sensitive_wins"
    elif c > a and b > d:
        game = "coexistence"
    elif a > c and d > b:
        game = "bistability"
    elif c > a and d > b:
        game = "resistant_wins"
    else:
        game = "unknown"
    return game


def test_fitting(data_type, source, *sample_ids):
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
    elif len(sys.argv) > 3:
        test_fitting(sys.argv[1], sys.argv[2], *sys.argv[3:])
    else:
        print("Please provide the data type.")