import os
import pickle
import sys

from spatial_egt.classification.common import df_to_xy, get_feature_data
from spatial_egt.classification.model_eval_utils import plot_confusion_matrix
from spatial_egt.common import get_data_path


def test_model(model_loc, save_loc, X, y, int_to_name):
    with open(f"{model_loc}/model.pkl", "rb") as f:
        clf = pickle.load(f)
    y_pred = clf.predict(X)
    plot_confusion_matrix(save_loc, "test", int_to_name, [y], [y_pred])


def main(model_data_type, data_type, label_name, feature_names):
    save_loc, df, feature_names = get_feature_data(data_type, label_name, feature_names)
    feature_df = df[feature_names+[label_name]]
    X, y, int_to_class = df_to_xy(feature_df, feature_names, label_name)
    model_features = os.path.basename(os.path.normpath(save_loc))
    model_loc = get_data_path(model_data_type, f"images/model/{model_features}")
    test_model(model_loc, save_loc, X, y, int_to_class)


if __name__ == "__main__":
    if len(sys.argv) > 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        print("Please provide the training data type, testing data type, label name, and feature set/names.")
