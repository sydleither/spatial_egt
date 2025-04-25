import sys

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatial_egt.common import game_colors, get_data_path


def downsample(s_coords, r_coords, ideal_size=(25,20)):
    dims = range(len(s_coords[0]))
    if isinstance(ideal_size, int):
        ideal_size = tuple([ideal_size]*len(dims))
    min_dims = tuple([min(np.min(s_coords[:, i]), np.min(r_coords[:, i])) for i in dims])
    max_dims = tuple([max(np.max(s_coords[:, i]), np.max(r_coords[:, i]))+1 for i in dims])
    sample_length = [(max_dims[i]-min_dims[i])//ideal_size[i] for i in dims]

    print(min_dims)
    print(max_dims)
    print(sample_length)

    grid = []
    curr_dims = min_dims
    while np.all(np.array(curr_dims) < np.array(max_dims)):
        ld = curr_dims
        ud = [ld[i]+sample_length[i] for i in dims]
        subset_s = [(s_coords[:, i] >= ld[i]) & (s_coords[:, i] <= ud[i]) for i in dims]
        subset_s = np.sum(np.all(subset_s, axis=0))
        subset_r = [(r_coords[:, i] >= ld[i]) & (r_coords[:, i] <= ud[i]) for i in dims]
        subset_r = np.sum(np.all(subset_r, axis=0))

    # grid = []
    # for s in range(len(dim_vals)):
    #     ld = [dim_vals[i][s] for i in dims]
    #     ud = [ld[i]+sample_length[i] for i in dims]
    #     subset_s = [(s_coords[:, i] >= ld[i]) & (s_coords[:, i] <= ud[i]) for i in dims]
    #     subset_s = np.sum(np.all(subset_s, axis=0))
    #     subset_r = [(r_coords[:, i] >= ld[i]) & (r_coords[:, i] <= ud[i]) for i in dims]
    #     subset_r = np.sum(np.all(subset_r, axis=0))
    #     subset_total = subset_s + subset_r
    #     if subset_total == 0:
    #         continue
    #     fs_counts.append(subset_s/subset_total)
    # return grid


def plot_sample(df, save_loc, sample_id):
    dimensions = list(df.drop("type", axis=1).columns)
    s_coords = df.loc[df["type"] == "sensitive"][dimensions].values
    r_coords = df.loc[df["type"] == "resistant"][dimensions].values
    #grid = downsample(s_coords, r_coords)

    color_map = {"sensitive":1, "resistant":2}
    df["color"] = df["type"].map(color_map)
    max_x = df["x"].max() + 1
    max_y = df["y"].max() + 1
    grid = np.zeros((max_y, max_x))
    for _, row in df.iterrows():
        grid[row["y"], row["x"]] = row["color"]

    scale = 4
    scaled_grid = np.kron(grid, np.ones((scale, scale)))
    fig, ax = plt.subplots(figsize=(scale*max_x, scale*max_y), dpi=1)
    cell_colors = [game_colors["Sensitive Wins"],
                   game_colors["Resistant Wins"]]
    cmap = ListedColormap(["#F0F0F0"]+cell_colors)
    plt.imshow(scaled_grid, cmap=cmap, interpolation="none")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    fig.tight_layout()
    plt.savefig(f"{save_loc}/sample_{sample_id}.png", bbox_inches="tight")


def main(data_type, source, *sample_ids):
    processed_data_path = get_data_path(data_type, "processed")
    image_data_path = get_data_path(data_type, "images")
    for sample_id in sample_ids:
        file_name = f"{source} {sample_id}.csv"
        df_sample = pd.read_csv(f"{processed_data_path}/{file_name}")
        plot_sample(df_sample, image_data_path, sample_id)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], *sys.argv[3:])
    else:
        print("Please provide the data type, source, and sample ids to plot.")
