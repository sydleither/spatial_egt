import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
import seaborn as sns

from common import game_colors, get_data_path
from data_analysis.plot_function import get_data
from data_processing.write_feature_jobs import DISTRIBUTION_BINS


def bin_samples(function_name, samples):
    start, stop, step = DISTRIBUTION_BINS[function_name]
    bins = np.arange(start, stop, step)
    binned_samples = []
    for data, game in samples:
        binned_data, _ = np.histogram(data, bins=bins)
        binned_samples.append([binned_data, game])
    return binned_samples


def get_difference_function(function1, function2):
    return euclidean(function1, function2)


def get_difference_distribution(distribution1, distribution2):
    #return euclidean(distribution1, distribution2)
    return wasserstein_distance(distribution1, distribution2)


def get_difference_feature(feature1, feature2):
    return abs(feature1 - feature2)


def get_difference_data(data_type, feature_name):
    features_data_path = get_data_path(data_type, "features")
    df_features = pd.read_csv(f"{features_data_path}/all.csv")
    df_features = df_features[df_features["game"] != "Unknown"]

    if feature_name in df_features.columns:
        samples = df_features[[feature_name, "game"]].values
        get_difference = get_difference_feature
        stat = "Absolute Difference"
    else:
        df_func = pd.read_pickle(f"{features_data_path}/{feature_name}.pkl")
        samples = get_data(df_func, feature_name, data_type)[[feature_name, "game"]].values
        if df_func["type"].iloc[0] == "distribution":
            samples = bin_samples(feature_name, samples)
            get_difference = get_difference_distribution
            stat = "Wasserstein Distance"
            #stat = "Euclidean Distance"
        else:
            get_difference = get_difference_function
            stat = "Euclidean Distance"

    distance_matrix = [[0 for _ in range(len(samples))] for _ in range(len(samples))]
    labels = []
    for i in range(len(samples)):
        feature_i, game_i = samples[i]
        labels.append(game_i)
        for j in range(i+1, len(samples)):
            feature_j, _ = samples[j]
            diff = int(get_difference(feature_i, feature_j))
            distance_matrix[i][j] = diff
            distance_matrix[j][i] = diff

    return np.array(distance_matrix), labels, stat


def plot_distances(save_loc, distance_matrix):
    fig, ax = plt.subplots()
    sns.heatmap(data=distance_matrix, cmap="Purples", ax=ax)
    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(f"{save_loc}/distance_matrix.png", bbox_inches="tight")


def run_knn(distance_matrix, labels):
    knn = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
    knn.fit(distance_matrix, labels)
    cv_predictions = cross_val_predict(knn, distance_matrix, labels, cv=5)

    cm = confusion_matrix(labels, cv_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(labels),
                yticklabels=np.unique(labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(labels, cv_predictions))


def main(data_type, feature_name):
    save_loc = get_data_path(data_type, f"images/{feature_name}")
    distance_matrix, labels, stat = get_difference_data(data_type, feature_name)
    run_knn(distance_matrix, labels)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide the data type and feature name.")
