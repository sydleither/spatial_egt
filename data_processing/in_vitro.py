import json
import os
import sys

import numpy as np
import pandas as pd

from spatial_statistics import create_all_features


def calculate_game(df):
    df["a"] = -1.0
    df["b"] = -1.0
    df["c"] = -1.0
    df["d"] = -1.0
    for drugcon in df["DrugConcentration"].unique():
        df_d = df.loc[df["DrugConcentration"] == drugcon]
        a = 0
        b = 0
        c = 0
        d = 0
        for cell_type in ["sensitive", "resistant"]:
            df_dc = df_d.loc[df_d["CellType"] == cell_type]
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
    cell_type_map = {"S-3E9": "sensitive", "BRAF-mCherry":"resistant",
                     "S-NLS": "sensitive", "R-NLS": "resistant"}
    raw_data_path = "data/in_vitro/raw/"
    for file_name in os.listdir(raw_data_path):
        file_name_parts = file_name.split("_")
        if file_name_parts[0] == "growth":
            df_gr = pd.read_csv(f"{raw_data_path}/{file_name}")
            df_gr = df_gr[["WellId", "CellType", "PlateId", 
                           "DrugConcentration", "Fraction_Sensitive",
                           "GrowthRate", "Intercept"]]
            df_gr = df_gr.drop_duplicates().dropna()
            df_gr["CellType"] = df_gr["CellType"].map(cell_type_map)
            df_gr = calculate_game(df_gr)
            df_gr = df_gr[["WellId", "PlateId", "a", "b", "c", "d"]]
            df_gr = df_gr.drop_duplicates().reset_index(drop=True)
            print(df_gr)
            exit()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "raw":
            raw_to_processed()
        elif sys.argv[1] == "processed":
            pass
        else:
            print("Data transformation options: raw or processed.")
    else:
        print("Please provide whether to convert raw data into processed (raw)")
        print("or processed data into features (processed).")