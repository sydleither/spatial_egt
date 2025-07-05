import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spatial_egt.classification.common import df_to_xy, get_feature_data
from spatial_egt.common import theme_colors


def finetune_layers(save_loc, X, y):
    cv = 5
    hidden_layer_sizes = [(i,) for i in range(50, 550, 50)]
    params = {"clf__hidden_layer_sizes":hidden_layer_sizes}

    pipeline = Pipeline([("scale", StandardScaler()), ("clf", MLPClassifier(max_iter=5000, solver="adam"))])
    search = GridSearchCV(pipeline, params, cv=cv).fit(X, y)

    df = pd.DataFrame(search.cv_results_)
    df = pd.melt(df, id_vars="param_clf__hidden_layer_sizes",
                 value_vars=[f"split{i}_test_score" for i in range(cv)])
    df["param_clf__hidden_layer_sizes"] = df["param_clf__hidden_layer_sizes"].astype(str).str[1:-2]
    df["param_clf__hidden_layer_sizes"] = df["param_clf__hidden_layer_sizes"].astype(int)
    df = df.rename({"param_clf__hidden_layer_sizes":"Number of Nodes", "value":"Accuracy"}, axis=1)

    print(df[["Number of Nodes", "Accuracy"]].groupby("Number of Nodes").mean().to_string())

    fig, ax = plt.subplots()
    sns.lineplot(df, x="Number of Nodes", y="Accuracy", color=theme_colors[0], ax=ax)
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.savefig(f"{save_loc}/layersize_tuning.png", bbox_inches="tight", dpi=200)


def main(data_type, label_name, feature_names):
    save_loc, df, feature_names = get_feature_data(data_type, label_name, feature_names)
    feature_df = df[feature_names+[label_name]]
    X, y, _ = df_to_xy(feature_df, feature_names, label_name)
    finetune_layers(save_loc, X, y)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
    else:
        print("Please provide the data type, label name, and feature set/names.")
