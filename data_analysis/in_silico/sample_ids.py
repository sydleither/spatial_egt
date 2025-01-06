import pandas as pd

from common.common import get_data_path
from data_processing.spatial_statistics import calculate_game


def main():
    processed_data_path = get_data_path("in_silico", "processed")
    df = pd.read_csv(f"{processed_data_path}/payoff.csv")
    df["sample"] = df["sample"].astype(str)
    df["game"] = df.apply(calculate_game, axis="columns")
    df = df[df["game"] != "unknown"]
    for game in df["game"].unique():
        print(game)
        df_game = df[df["game"] == game]
        df_game_clean = df_game[(df_game["initial_fr"] <= 0.55) &
                                (df_game["initial_fr"] >= 0.45) &
                                (df_game["initial_density"] <= 8812) &
                                (df_game["initial_density"] >= 6812)]
        sample_ids = df_game_clean["sample"].values
        print(" ".join(sample_ids))
        print()


if __name__ == "__main__":
    main()