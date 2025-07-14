import os
import sys

import matplotlib.pyplot as plt

from spatial_egt.classification.common import get_feature_data, read_in_data
from spatial_egt.common import theme_colors

import warnings

warnings.filterwarnings("ignore")


def plot_score_by_features(save_loc):
    num_features = []
    scores = []
    for result in os.listdir(save_loc):
        if not result.endswith(".csv"):
            continue
        previous_results = open(f"{save_loc}/{result}", encoding="UTF-8").read()
        previous_results = [x.split(",") for x in previous_results.split("\n")]
        scores.append(float(previous_results[1][-1]))
        num_features.append(int(result[:-4]))
    num_features, scores = zip(*sorted(zip(num_features, scores)))

    fig, ax = plt.subplots()
    ax.plot(num_features, scores, c=theme_colors[0], marker="o")
    ax.set(
        xlabel="Number of Features in Top-Performing Feature Set",
        ylabel="Mean 5-Fold CV Accuracy",
        title="Classifier Accuracy by Number of Features",
    )
    fig.tight_layout()
    fig.figure.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/accuracy_by_features.png", bbox_inches="tight", dpi=200)
    plt.close()


def main(args):
    data_type, time, label_name, feature_names = read_in_data(args)
    save_loc, _, _ = get_feature_data(
        data_type, time, label_name, feature_names, "sfs"
    )
    plot_score_by_features(save_loc)


if __name__ == "__main__":
    main(sys.argv)
