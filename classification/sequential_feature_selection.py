import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold

from classification.common import df_to_xy, read_and_clean_features


def sfs(X, y, feature_names): #TODO add standard scalar
    feature_names = np.array(feature_names)
    cv = StratifiedKFold(5)
    rf = RandomForestClassifier()
    sfs_forward = SequentialFeatureSelector(rf, tol=0.05, direction="forward", cv=cv)
    sfs_forward.fit(X, y)
    print("Forward:", feature_names[sfs_forward.get_support()])
    sfs_backward = SequentialFeatureSelector(rf, tol=-0.05, direction="backward", cv=cv)
    sfs_backward.fit(X, y)
    print("Backward:", feature_names[sfs_backward.get_support()])


def main(experiment_name, *data_types):
    label = ["game"]
    feature_df = read_and_clean_features(data_types[0], label, experiment_name)
    X, y, _, feature_names = df_to_xy(feature_df, label[0])
    sfs(X, y, feature_names)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide a feature set and the data types to train the model with.")
