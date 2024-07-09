import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def read_payoffs():
    rows = []
    exp_path = "output/gamespc_all"
    for exp_name in os.listdir(exp_path):
        run_path = f"{exp_path}/{exp_name}"
        if os.path.isfile(run_path):
            continue
        for rep_dir in os.listdir(run_path):
            rep_path = f"{run_path}/{rep_dir}"
            if rep_path.endswith(".json"):
                row = {}
                config = json.load(open(rep_path))
                row["A"] = config["A"]
                row["B"] = config["B"]
                row["C"] = config["C"]
                row["D"] = config["D"]
                row["exp_name"] = exp_name
                exp_name_split = exp_name.split("_")
                row["game"] = exp_name_split[0]
                row["beta"] = exp_name_split[1]
                rows.append(row)
    df = pd.DataFrame(rows)
    df["c-a"] = df["C"] - df["A"]
    df["b-d"] = df["B"] - df["D"]
    return df


def plot_gamespace(df, transparent=False):
    colors = ["sienna", "hotpink", "limegreen", "royalblue"]
    offset = {"sensitive":(-15,0), "coexistence":(0,0), "bistability":(-10,-10), "resistant":(0,-15)}
    fig, ax = plt.subplots()
    for i,game in enumerate(df["game"].unique()):
        df_game = df.loc[df["game"] == game]
        xs = df_game["c-a"].values
        ys = df_game["b-d"].values
        betas = df_game["beta"].values
        ax.scatter(xs, ys, label=game, color=colors[i])
        for j in range(len(xs)):
            ax.annotate(text=betas[j], xy=(xs[j], ys[j]), xytext=offset[game], textcoords="offset points")
    ax.axhline(color="black")
    ax.axvline(color="black")
    ax.set(xlim=(-0.1, 0.1), ylim=(-0.015, 0.015), title="Game Space")
    ax.set_xlabel("Relative Fitness of Resistant")
    ax.set_ylabel("Relative Fitness of Sensitive")
    ax.legend()
    fig.tight_layout()
    if transparent:
        fig.patch.set_alpha(0.0)
    fig.savefig("output/gamespc_all/gamespace.png")


def main():
    df = read_payoffs()
    plot_gamespace(df)


if __name__ == "__main__":
    main()