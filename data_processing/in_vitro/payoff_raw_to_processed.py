import numpy as np
import pandas as pd

from common import (cell_type_map, experiment_names,
                    make_data_dirs, processed_data_path, 
                    raw_data_path)


def calculate_starting_fs(df):
    df_sum = df.groupby(["PlateId", "WellId"])["Count"].sum()
    df_sum = df_sum.reset_index()
    df_sum = df_sum.rename(columns={"Count":"TotalCells"})
    df_sens = df[df["CellType"] == "sensitive"]
    df_sens = df_sum.merge(df_sens, on=["PlateId", "WellId"])
    df_sens["FractionSensitive"] = df_sens["Count"]/df_sens["TotalCells"]
    df_sens = df_sens[["PlateId", "WellId", "FractionSensitive"]]
    df = df.merge(df_sens, on=["PlateId", "WellId"])
    return df


def calculate_payoff(df):
    if len(df["PlateId"].unique()) > 1:
        print("Only calculate the payoff of one plate at a time.")
        exit()
    df["a"] = -1.0
    df["b"] = -1.0
    df["c"] = -1.0
    df["d"] = -1.0
    # DrugConcentration is used as condition id
    # Each drug concentration has multiple wells 
    # with differing starting sensitive fractions
    for drugcon in df["DrugConcentration"].unique():
        df_d = df[df["DrugConcentration"] == drugcon]
        a = 0
        b = 0
        c = 0
        d = 0
        # Game Assay (Farrokhian et al., 2021)
        for cell_type in ["sensitive", "resistant"]:
            df_dc = df_d[df_d["CellType"] == cell_type]
            growth_rate = df_dc["GrowthRate"].values
            fraction_s = df_dc["FractionSensitive"].values
            slope, intercept = np.polyfit(fraction_s, growth_rate, 1)
            if cell_type == "sensitive":
                a = slope+intercept
                b = intercept
            if cell_type == "resistant":
                c = slope+intercept
                d = intercept
        df.loc[df["DrugConcentration"] == drugcon, "a"] = a
        df.loc[df["DrugConcentration"] == drugcon, "b"] = b
        df.loc[df["DrugConcentration"] == drugcon, "c"] = c
        df.loc[df["DrugConcentration"] == drugcon, "d"] = d
    return df


def raw_to_processed():
    make_data_dirs()
    for experiment_name in experiment_names:
        # Process counts file, which is necessary for game assay,
        # since the game assay uses the initial proportion sensitive
        counts_name = f"counts_df_processed_{experiment_name}_plate1.csv"
        df_counts = pd.read_csv(f"{raw_data_path}/{counts_name}")
        df_counts = df_counts.loc[df_counts["Time"] == 0]
        df_counts["CellType"] = df_counts["CellType"].map(cell_type_map)
        df_counts = df_counts[["PlateId", "WellId", "CellType", "Count"]]

        # Process growth rate file to calculate payoff matrix
        growth_name = f"growth_rate_df_{experiment_name}_plate1.csv"
        df_gr = pd.read_csv(f"{raw_data_path}/{growth_name}")
        df_gr = df_gr[["PlateId", "WellId", "CellType",
                       "DrugConcentration", "GrowthRate", "Intercept"]]
        df_gr["CellType"] = df_gr["CellType"].map(cell_type_map)
        df_gr = df_gr.merge(df_counts, on=["PlateId", "WellId", "CellType"])
        df_gr = calculate_starting_fs(df_gr).dropna()
        df_gr = calculate_payoff(df_gr)
        df_gr = df_gr[["PlateId", "WellId",
                       "DrugConcentration",
                       "a", "b", "c", "d"]]
        df_gr = df_gr.drop_duplicates().reset_index(drop=True)

        # Save processed payoff dataframe
        df_gr.to_csv(f"{processed_data_path}/payoff_{experiment_name}.csv", index=False)


if __name__ == "__main__":
    raw_to_processed()