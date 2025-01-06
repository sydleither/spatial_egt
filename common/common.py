import os

dimension = "2D"
in_vitro_exp_names = ["braf", "nls"]
cell_type_map = {0: "sensitive", 1:"resistant",
                 "S-3E9": "sensitive", "BRAF-mCherry":"resistant",
                 "S-NLS": "sensitive", "R-NLS": "resistant",
                 "mCherry": "resistant"}
game_colors = {"sensitive_wins":"#4C956C", "coexistence":"#F97306",
               "bistability":"#047495", "resistant_wins":"#EF7C8E"}


def get_data_path(data_type, data_stage):
    data_path = f"data/{data_type}/{data_stage}"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path