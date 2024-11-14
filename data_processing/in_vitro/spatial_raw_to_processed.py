import pandas as pd

from common import (cell_type_map, get_data_path, in_vitro_exp_names)


def raw_to_processed():
    raw_data_path = get_data_path("in_vitro", "raw")
    processed_data_path = get_data_path("in_vitro", "processed")

    for experiment_name in in_vitro_exp_names:
        # Get latest time
        spatial_name = f"locations_df_processed_{experiment_name}_plate1.csv"
        max_time = 0
        with open(f"{raw_data_path}/{spatial_name}") as f:
            next(f)
            for line in f:
                time = int(line.split(",")[1])
                if time > max_time:
                    max_time = time

        # Read in rows of spatial file with latest time
        header = ""
        rows = []
        with open(f"{raw_data_path}/{spatial_name}") as f:
            header = next(f)
            for line in f:
                time = int(line.split(",")[1])
                if time == max_time:
                    rows.append(line.strip().split(","))
        header = header.strip().split(",")

        # Convert to dataframe
        df = pd.DataFrame(data=rows, columns=header)
        df = df.rename(columns={"Location_Center_X":"x",
                                "Location_Center_Y":"y",
                                "CellType": "type"})
        df = df[["PlateId", "WellId", "type", "x", "y"]]
        df["type"] = df["type"].map(cell_type_map)
        df["x"] = df["x"].astype(float).round(0).astype(int)
        df["y"] = df["y"].astype(float).round(0).astype(int)

        # Save wells separately
        for well in df["WellId"].unique():
            df_well = df[df["WellId"] == well][["type", "x", "y"]]
            df_well.to_csv(f"{processed_data_path}/spatial_{experiment_name}_{well}.csv", index=False)


if __name__ == "__main__":
    raw_to_processed()