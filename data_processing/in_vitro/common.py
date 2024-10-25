import os

raw_data_path = "data/in_vitro/raw"
processed_data_path = "data/in_vitro/processed"

experiment_names = ["braf", "nls"]

cell_type_map = {"S-3E9": "sensitive", "BRAF-mCherry":"resistant",
                 "S-NLS": "sensitive", "R-NLS": "resistant",
                 "mCherry": "resistant"}

def make_data_dirs():
    if not os.path.exists(raw_data_path):
        print(f"Please move the raw in vitro data to {raw_data_path}.")
        os.makedirs(raw_data_path)
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)