import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import stats

from common import get_data_path
from data_processing.processed_to_features import read_processed_sample
from data_processing.spatial_statistics import (calculate_game, create_sfp_dist)


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


def classify(data_type):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = get_payoff_data(processed_data_path)
    samples = list(df_payoff[["sample", "source", "game"]].values)

    dists, games = get_theoretical_dists(1000)

    y_pred = []
    y_true = []
    for sample_id, source, game in samples:
        sample_dist = get_sfp_dist(processed_data_path, df_payoff, data_type, 
                                   source, sample_id, None, False)
        scores = []
        for i in range(len(dists)):
            score = stats.wasserstein_distance(sample_dist, dists[i])
            scores.append(score)
        classification = games[np.argmin(scores)]
        y_pred.append(classification)
        y_true.append(game)
    acc = sum([1 for i in range(len(y_pred)) if y_pred[i] == y_true[i]])/len(y_pred)

    save_loc = get_data_path(data_type, "images")
    conf_mat = confusion_matrix(y_true, y_pred, labels=games)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=games)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    ax.set_title(f"Accuracy: {acc}")
    fig.tight_layout()
    fig.savefig(f"{save_loc}/em_confusion_matrix.png", bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        classify(sys.argv[1])
    else:
        print("Please provide the data type.")