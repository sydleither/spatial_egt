from common import get_data_path, read_payoff_df


def main():
    processed_data_path = get_data_path("in_vitro", "processed")
    df_payoff = read_payoff_df(processed_data_path)
    for exp_name in df_payoff["source"].unique():
        print(exp_name)
        exp_samples = df_payoff[df_payoff["source"] == exp_name]
        for drugcon in sorted(exp_samples["DrugConcentration"].unique()):
            game = exp_samples[exp_samples["DrugConcentration"] == drugcon]["game"].iloc[0]
            print("\t", drugcon, game)


if __name__ == "__main__":
    main()