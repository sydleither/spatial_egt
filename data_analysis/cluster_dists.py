import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import mutual_info_score, adjusted_rand_score
from scipy import stats
import seaborn as sns

from common.common import game_colors, get_data_path
from common.distributions import (get_payoff_data, 
                                  get_sfp_dist)


def main(data_type, distribution_name):
    processed_data_path = get_data_path(data_type, "processed")
    df_payoff = get_payoff_data(processed_data_path)
    df_payoff = df_payoff.set_index("sample", drop=False)
    sample_ids = list(df_payoff["sample"].values)[0:100]
    num_samples = len(sample_ids)

    distance_matrix = [[0 for _ in range(num_samples)] for _ in range(num_samples)]
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            sample_id1 = sample_ids[i]
            sample_id2 = sample_ids[j]
            sample_data1 = df_payoff.loc[[sample_id1]]
            sample_data2 = df_payoff.loc[[sample_id2]]
            sample_dist1 = get_sfp_dist(processed_data_path, df_payoff,
                                        data_type, sample_data1["source"][0],
                                        sample_id1, 5, False)
            sample_dist2 = get_sfp_dist(processed_data_path, df_payoff,
                                        data_type, sample_data2["source"][0],
                                        sample_id2, 5, False)
            score = stats.wasserstein_distance(sample_dist1, sample_dist2)
            distance_matrix[i][j] = score
            distance_matrix[j][i] = score

    max_distance = max([max(row) for row in distance_matrix])
    simularity_matrix = [[max_distance-x for x in row] for row in distance_matrix]
    clustering = SpectralClustering(n_clusters=4, affinity="precomputed")
    cluster_labels = clustering.fit_predict(simularity_matrix)
    sample_labels = list(df_payoff["game"].values)[0:100]

    ari = adjusted_rand_score(sample_labels, cluster_labels)
    mi = mutual_info_score(sample_labels, cluster_labels)
    print(f"Adjusted Rand index: {ari:5.3f}\nMutual Information: {mi:5.3f}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the data type and distribution name (options: sfp or nc).")