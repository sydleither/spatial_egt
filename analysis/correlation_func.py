import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

from common import read_all
from common import read_all, get_colors
from spatial_statistics import model_state_to_coords, read_model_state


def plot_func(xs, ys, labels, exp_dir, dimension, name, colors, transparent=False):
    fig, ax = plt.subplots()
    for i in range(len(xs)):
        x_vals = xs[i]
        y_vals = ys[i]
        label = labels[i]
        color = colors[i]
        ax.plot(x_vals, y_vals, label=label, color=color)
    ax.set_xlabel("Distance Between Pairs of Cells")
    ax.set_ylabel("Correlation Function Signal")
    ax.set_title(name)
    ax.legend()
    fig.tight_layout()
    if transparent:
        fig.patch.set_alpha(0.0)
    fig.savefig(f"output/{exp_dir}/{name}.png")


def count_manhattan_pairs(n, m, r):
    total_pairs = 0
    # Iterate over all possible dx values (horizontal separation)
    for dx in range(r + 1):
        # dy is determined by the equation dx + dy = r
        dy = r - dx
        # Number of valid rows and columns for each (dx, dy)
        valid_rows = max(0, n - dy)  # Number of rows that can support vertical separation dy
        valid_columns = max(0, m - dx)  # Number of columns that can support horizontal separation dx
        # Multiply valid rows and columns to get the number of pairs for this (dx, dy)
        total_pairs += valid_rows * valid_columns
        # If dx != 0 and dy != 0, count reverse direction pairs only once
        if dx != 0 and dy != 0:
            total_pairs += valid_rows * valid_columns
    return total_pairs//2


def correlation2(model_state):
    '''
    manhattan
    0 -0.007232
    1 -0.5088064516129033
    2 -0.5056908718415661
    3 -0.5076496044061721
    euclidean
    0 -0.006243496357960458 3820 3844
    1 -0.5025989604158336 14928 30012
    2 -0.5117827868852459 28590 58560
    '''
    cr = dict()
    grid = 62
    for r in range(62):
        count = 0
        norm = 0
        for i in range(grid):
            for j in range(grid):
                for k in range(grid):
                    for l in range(grid):
                        celli = model_state[i][j]
                        cellk = model_state[k][l]
                        if int(distance.euclidean((i,j),(k,l))) == r:
                            norm += 1
                            if celli == cellk and celli != "*":
                                count += 1
        cr[r] = (count-norm)/norm
        print(r, cr[r], count, norm)
    return cr


def correlation(s_coords, r_coords):
    all_coords = [(i,"s",s_coords[i]) for i in range(len(s_coords))]
    all_coords += [(i+len(s_coords),"r",r_coords[i]) for i in range(len(r_coords))]

    num_pairs = {r:count_manhattan_pairs(125,125,r) for r in range(126)}

    cr = {r:0 for r in range(126)}
    for i in range(len(all_coords)):
        for j in range(i+1, len(all_coords)):
            c1_idx, c1_type, c1_coords = all_coords[i]
            c2_idx, c2_type, c2_coords = all_coords[j]
            dist = abs(c1_coords[0] - c2_coords[0]) + abs(c1_coords[1] - c2_coords[1])
            if dist > 125:
                continue
            if c1_type == c2_type:
                cr[dist] += 1
    cr = {r:(cr[r]-num_pairs[r])/num_pairs[r] for r in range(1,126)}

    return cr


def normalize_pc(radius, count, pair, num_s, num_r):
    if radius == 125:
        return 0
    if pair == "ss":
        return count/(15625*(125-radius)*(num_s/15625)*((num_s-1)/(15624)))
    elif pair == "rr":
        return count/(15625*(125-radius)*(num_r/15625)*((num_r-1)/(15624)))
    else:
        return count/(15625*(125-radius)*(((num_s/15625)*((num_r)/(15624)))+((num_r/15625)*((num_s)/(15624)))))


def pair_correlation(s_coords, r_coords):
    all_coords = [(i,"s",s_coords[i]) for i in range(len(s_coords))]
    all_coords += [(i+len(s_coords),"r",r_coords[i]) for i in range(len(r_coords))]

    pc_funcs_x = {pair:{r:0 for r in range(126)} for pair in ["ss", "sr", "rr"]}
    pc_funcs_y = {pair:{r:0 for r in range(126)} for pair in ["ss", "sr", "rr"]}
    for i in range(len(all_coords)):
        for j in range(i+1, len(all_coords)):
            c1_idx, c1_type, c1_coords = all_coords[i]
            c2_idx, c2_type, c2_coords = all_coords[j]
            xdist = abs(c1_coords[0] - c2_coords[0])
            ydist = abs(c1_coords[1] - c2_coords[1])
            pair = c1_type+c2_type
            if pair == "rs":
                continue
            pc_funcs_x[pair][xdist] += 1
            pc_funcs_y[pair][ydist] += 1

    num_s = len(s_coords)
    num_r = len(r_coords)
    for pair in ["ss", "sr", "rr"]:
        pc_funcs_x[pair] = {r:normalize_pc(r, v, pair, num_s, num_r) for r,v in pc_funcs_x[pair].items()}
        pc_funcs_y[pair] = {r:normalize_pc(r, v, pair, num_s, num_r) for r,v in pc_funcs_y[pair].items()}

    return pc_funcs_x, pc_funcs_y


def main(exp_dir, dimension):
    #bull pair correlations
    df_pc = read_all(exp_dir, "pairCorrelations", dimension)
    df_pop = read_all(exp_dir, "populations", dimension)
    df = df_pop.merge(df_pc, on=["model", "time", "rep", "condition", "dimension"])
    df["pc"] = df["normalized_count"] / (df["sensitive"]*df["resistant"])

    bpc_xs = []
    bpc_ys = []
    bpc_labels = []
    for pair in df["pair"].unique():
        df_pair = df.loc[(df["pair"] == pair)]
        bpc_xs.append(list(df_pair["radius"].values))
        bpc_ys.append(list(df_pair["pc"].values))
        bpc_labels.append(pair)
    plot_func(bpc_xs, bpc_ys, bpc_labels, exp_dir, dimension, "Bull Pair Correlation", get_colors())

    #pair correlations
    model = read_model_state(f"output/{exp_dir}/test/0/{dimension}model1.csv")
    s_coords, r_coords = model_state_to_coords(model)
    pc_funcs_x, pc_funcs_y = pair_correlation(s_coords, r_coords)

    pc_xs = []
    pc_ys = []
    pc_labels = []
    for pair in set(pc_funcs_x.keys()):
        pc_ys.append(list(pc_funcs_x[pair].values()))
        pc_ys.append(list(pc_funcs_y[pair].values()))
        pc_xs.append(list(pc_funcs_x[pair].keys()))
        pc_xs.append(list(pc_funcs_y[pair].keys()))
        pc_labels.append(f"{pair}_x")
        pc_labels.append(f"{pair}_y")
    colors = ["lightgreen", "limegreen", "sandybrown", "chocolate",
              "cyan", "darkturquoise", "orchid", "mediumorchid"]
    plot_func(pc_xs, pc_ys, pc_labels, exp_dir, dimension, "Pair Correlation", colors)

    #correlations
    cr = correlation(s_coords, r_coords)
    c_xs = [list(cr.keys())]
    c_ys = [list(cr.values())]
    c_labels = ["manhattan"]
    plot_func(c_xs, c_ys, c_labels, exp_dir, dimension, "Correlation", get_colors())


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide an experiment directory and dimension.")