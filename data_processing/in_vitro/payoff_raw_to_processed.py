import numpy as np
import pandas as pd

from common import (cell_type_map, experiment_names,
                    make_data_dirs, processed_data_path, 
                    raw_data_path)


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
            fraction_s = df_dc["Fraction_Sensitive"].values
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
        # Process growth rate file to calculate payoff matrix
        growth_name = f"growth_rate_df_{experiment_name}_plate1.csv"
        df_gr = pd.read_csv(f"{raw_data_path}/{growth_name}")
        df_gr = df_gr[["PlateId", "WellId", "CellType", "Fraction_Sensitive",
                       "DrugConcentration", "GrowthRate", "Intercept"]]
        df_gr["CellType"] = df_gr["CellType"].map(cell_type_map)
        df_gr = df_gr.dropna()
        df_gr = calculate_payoff(df_gr)
        df_gr = df_gr[["PlateId", "WellId",
                       "a", "b", "c", "d"]]
        df_gr = df_gr.drop_duplicates().reset_index(drop=True)
        # Save processed payoff dataframe
        df_gr.to_csv(f"{processed_data_path}/payoff_{experiment_name}.csv", index=False)


if __name__ == "__main__":
    raw_to_processed()