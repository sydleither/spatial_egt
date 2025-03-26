from itertools import combinations
import sys

import numpy as np
import pandas as pd

from classification.common import read_and_clean_features
from classification.DDIT.DDIT import DDIT
from common import get_data_path


def fragmentation_data(save_loc, df, label_name, feature_set_size):
    #initializations
    ddit = DDIT()
    feature_names = list(df.columns)
    feature_names.remove(label_name)
    feature_name_map = {name:str(i) for i,name in enumerate(feature_names)}

    #bin and register features
    nbins = int(np.log2(len(df)))+1
    for feature_name in feature_names:
        feature_name_index = feature_name_map[feature_name]
        column_data = df[feature_name].values
        binned_column_data = pd.qcut(column_data, nbins, labels=False)
        ddit.register_column_tuple(feature_name_index, tuple(binned_column_data))
    ddit.register_column_tuple(label_name, tuple(df[label_name].values))

    #calculate entropies
    feature_sets = combinations(feature_name_map.values(), feature_set_size)
    label_entropy = ddit.H(label_name)
    results = []
    for feature_set in feature_sets:
        ent = ddit.recursively_solve_formula(label_name+":"+"&".join(feature_set)) / label_entropy
        results.append([feature_names[int(i)] for i in feature_set]+[f"{ent:5.3f}"])

    #save
    with open(f"{save_loc}/{feature_set_size}.csv", "w") as f:
        f.write(",".join([str(i) for i in range(feature_set_size)])+","+"value\n")
        for result in results:
            f.write(",".join(result)+"\n")


def main(experiment_name, feature_set_size, data_type):
    label = ["game"]
    feature_df = read_and_clean_features([data_type], label, experiment_name)
    save_loc = get_data_path(data_type, f"model/{experiment_name}/fragmentation")
    fragmentation_data(save_loc, feature_df, label[0], int(feature_set_size))


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Please provide a feature set, size, and the data type.")