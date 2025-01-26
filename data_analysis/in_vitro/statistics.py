import pandas as pd

from common import get_data_path
from data_processing.spatial_statistics import calculate_game


def main():
    processed_data_path = get_data_path("in_vitro", "processed")
    df_payoff = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df_payoff["game"] = df_payoff.apply(calculate_game, axis="columns")
    for exp_name in df_payoff["source"].unique():
        print(exp_name)
        exp_samples = df_payoff[df_payoff["source"] == exp_name]
        for drugcon in sorted(exp_samples["DrugConcentration"].unique()):
            game = exp_samples[exp_samples["DrugConcentration"] == drugcon]["game"].iloc[0]
            print("\t", drugcon, game)


if __name__ == "__main__":
    main()