import pandas as pd

from common import (cell_colors, get_experiment_names,
                    make_image_dir, processed_data_path)


def main():
    for exp_name in get_experiment_names():
        make_image_dir(exp_name)
        df = pd.read_csv(f"{processed_data_path}/payoff_{exp_name}.csv")


if __name__ == "__main__":
    main()