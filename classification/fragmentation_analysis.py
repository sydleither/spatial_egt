import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import get_data_path


def main(feature_set, data_type):
    data_path = get_data_path(data_type, f"model/{feature_set}/fragmentation")
    df = pd.DataFrame()
    for file in os.listdir(data_path):
        if not file.endswith(".csv"):
            continue
        df_fs = pd.read_csv(f"{data_path}/{file}")
        cols = [x for x in df_fs.columns if x != "entropy"]
        df_fs["features"] = df_fs[cols].apply(lambda row: "\n".join(row.values.astype(str)), axis=1)
        df_fs["number"] = int(file[:-4])
        df = pd.concat([df, df_fs[["number", "features", "entropy"]]])

    for number in df["number"].unique():
        df_num = df[df["number"] == number]
        df_num_top = df_num.nlargest(10, "entropy")

        fig, ax = plt.subplots(figsize=(6, 15))
        sns.barplot(data=df_num_top, x="entropy", y="features", color="pink", ax=ax)
        ax.set(title=number)
        fig.tight_layout()
        fig.figure.patch.set_alpha(0.0)
        fig.savefig(f"{data_path}/{number}.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide a feature set and the data type.")
