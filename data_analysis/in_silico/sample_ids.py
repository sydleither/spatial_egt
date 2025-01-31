from common import get_data_path, read_payoff_df


def main():
    processed_data_path = get_data_path("in_silico", "processed")
    df = read_payoff_df(processed_data_path)
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