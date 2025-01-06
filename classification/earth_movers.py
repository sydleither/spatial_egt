import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import stats

from common.common import get_data_path
from common.distributions import get_payoff_data, get_sfp_dist, get_theoretical_dists


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