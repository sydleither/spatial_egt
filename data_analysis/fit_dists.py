import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common.common import game_colors, get_data_path
from common.distributions import (fit_beta, get_payoff_data, 
                                  get_sfp_dist)


def main(data_type, subsample_size):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = get_payoff_data(processed_data_path)
    samples = list(df_payoff[["sample", "source", "game"]].values)

    rows = []
    for sample_id, source, game in samples[0:1000]:
        sample_dist = get_sfp_dist(processed_data_path, df_payoff, data_type, 
                                   source, sample_id, int(subsample_size), False)
        a, b, mean, var = fit_beta(sample_dist)
        rows.append([sample_id, source, game, a, b, mean, var])

    df = pd.DataFrame(data=rows, columns=["sample", "source", "game",
                                          "a", "b", "mean", "var"])
    save_loc = get_data_path(data_type, "images")
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        game = list(game_colors.keys())[i]
        sns.histplot(data=df[df["game"] == game], x="a", y="b",
                    color=game_colors[game], ax=ax[i])
        ax[i].set(title=game, xlim=(0,6), ylim=(0,6))
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/dists_comparison_{subsample_size}.png")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the data type and subsample size.")