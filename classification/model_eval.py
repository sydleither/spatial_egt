import sys

from sklearn.model_selection import StratifiedKFold

from classification.common import get_model, read_and_clean_features
from classification.performance_plots import plot_all
from common import get_data_path


def cross_val(save_loc, X, y, int_to_name):
    all_y_train = []
    all_y_test = []
    all_y_train_pred = []
    all_y_test_pred = []
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True)
    for (train_i, test_i) in cross_validation.split(X, y):
        X_train = [X[i] for i in train_i]
        X_test = [X[i] for i in test_i]
        y_train = [y[i] for i in train_i]
        y_test = [y[i] for i in test_i]
        clf = get_model().fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        all_y_train.append(y_train)
        all_y_test.append(y_test)
        all_y_train_pred.append(y_train_pred.tolist())
        all_y_test_pred.append(y_test_pred.tolist())
    plot_all(save_loc, int_to_name, all_y_train, all_y_train_pred, "train")
    plot_all(save_loc, int_to_name, all_y_test, all_y_test_pred, "test")


def main(experiment_name, *data_types):
    save_loc = get_data_path(".", f"model/{experiment_name}")
    X, y, int_to_name = read_and_clean_features(data_types[0])
    cross_val(save_loc, X, y, int_to_name)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide an experiment name and the data types to train the model with.")
