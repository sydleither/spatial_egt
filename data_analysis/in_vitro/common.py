import os


raw_data_path = "data/in_vitro/raw"
processed_data_path = "data/in_vitro/processed"
cell_colors = ["#4C956C", "#EF7C8E"]

cell_type_map = {"S-3E9": "sensitive", "BRAF-mCherry":"resistant",
                 "S-NLS": "sensitive", "R-NLS": "resistant",
                 "mCherry": "resistant"}


def make_image_dir(exp_name):
    if not os.path.exists(f"data/in_vitro/images/{exp_name}"):
        os.makedirs(f"data/in_vitro/images/{exp_name}")


def get_experiment_names():
    exp_names = []
    for file in os.listdir(processed_data_path):
        if file.startswith("spatial"):
            exp_name = file.split("_")[-1][:-4]
            exp_names.append(exp_name)
    return exp_names