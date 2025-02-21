import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification.common import df_to_xy, read_and_clean_features
from common import get_data_path


def finetune_layers(save_loc, X, y):
    cv = 5
    hidden_layer_sizes = []
    for i in range(1, 6):
        layers = [100]*i
        hidden_layer_sizes.append(tuple(layers))
    hidden_layer_sizes.append((500,250,125,62))
    hidden_layer_sizes.append((500,400,300,200,100))
    hidden_layer_sizes.append((500,500,500))
    hidden_layer_sizes.append((200,100,50))
    params = {"clf__hidden_layer_sizes":hidden_layer_sizes}

    pipeline = Pipeline([("scale", StandardScaler()), ("clf", MLPClassifier(max_iter=500))])
    search = GridSearchCV(pipeline, params, cv=cv).fit(X, y)

    df = pd.DataFrame(search.cv_results_)
    df = pd.melt(df, id_vars="param_clf__hidden_layer_sizes",
                 value_vars=[f"split{i}_test_score" for i in range(cv)])
    df["param_clf__hidden_layer_sizes"] = df["param_clf__hidden_layer_sizes"].astype(str)

    fig, ax = plt.subplots()
    sns.barplot(df, x="param_clf__hidden_layer_sizes",
                y="value", color="hotpink", ax=ax)
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/hyperparameter_tuning.png", bbox_inches="tight")


def main(experiment_name, *data_types):
    parent_dir = "."
    if len(data_types[0]) == 1:
        parent_dir = data_types[0][0]
    save_loc = get_data_path(parent_dir, f"model/{experiment_name}")
    feature_df = read_and_clean_features(data_types[0], ["game"])
    X, y, _, _ = df_to_xy(feature_df, "game")
    finetune_layers(save_loc, X, y)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    else:
        print("Please provide an experiment name and the data types to train the model with.")
