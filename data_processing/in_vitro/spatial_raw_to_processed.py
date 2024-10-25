import pandas as pd

from common import (cell_type_map, experiment_names,
                    make_data_dirs, processed_data_path, 
                    raw_data_path)


def raw_to_processed():
    make_data_dirs()
    for experiment_name in experiment_names:
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
        df_sp = pd.DataFrame(data=rows, columns=header)
        df_sp = df_sp.rename(columns={"Location_Center_X":"x",
                                      "Location_Center_Y":"y",
                                      "CellType": "type"})
        df_sp = df_sp[["PlateId", "WellId", "DrugConcentration", "type", "x", "y"]]
        df_sp["type"] = df_sp["type"].map(cell_type_map)
        df_sp["x"] = df_sp["x"].astype(float).round(0).astype(int)
        df_sp["y"] = df_sp["y"].astype(float).round(0).astype(int)

        # Save processed spatial dataframe
        df_sp.to_csv(f"{processed_data_path}/spatial_{experiment_name}.csv", index=False)


if __name__ == "__main__":
    raw_to_processed()